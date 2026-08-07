// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![cfg_attr(all(target_os = "linux", target_env = "gnu"), no_main)]

//! Fuzzes the virtio-blk device by driving its request queue the way a guest
//! driver would: descriptor chains are written into guest memory, published in
//! the available ring, and the device is kicked through its queue event.
//!
//! Everything the device parses on this path is guest-controlled — the request
//! header, the split of a chain into device-readable and device-writable
//! descriptors, the GPAs and lengths of every descriptor, and the ring indices
//! themselves. The device is expected to survive any of it: a malformed chain
//! must be rejected (or completed with an error status byte), never panic,
//! never read or write outside the guest memory it was given, and never wedge
//! the worker so that queue teardown cannot complete.
//!
//! The grammar therefore mixes well-formed IN/OUT/FLUSH/GET_ID/DISCARD requests
//! with deliberately broken ones (short headers, missing status bytes, wrong
//! descriptor directions, zero-length and out-of-range descriptors, INDIRECT
//! flags (negotiated and not), unknown request types) and interleaves them with the queue
//! lifecycle operations a transport performs: kick, stop, reset, and restart
//! (both fresh and from the saved `QueueState`).
//!
//! Only queue 0 is ever touched: virtio-blk advertises `max_queues: 1` and
//! `start_queue`/`stop_queue` assert on any other index, so a queue index is
//! not a guest-controlled value and is not fuzzed.
//!
//! The negotiated feature word is also guest-controlled and is drawn from the
//! input (masked to what the device offers), except for `VIRTIO_F_RING_PACKED`,
//! which is always cleared because the rings this harness builds are split
//! rings.
//!
//! The harness and the device share a single-threaded [`DefaultPool`], so
//! execution is deterministic: the device only makes progress when the harness
//! yields, and every yield is charged against a budget. There is no sleeping
//! and no wall-clock wait anywhere in this target — in particular
//! `virtio::test_helpers::wait_for_used` is deliberately not used, because its
//! 5-second timeout would dominate the fuzzing campaign.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use guestmem::GuestMemory;
use pal_async::DefaultDriver;
use pal_async::DefaultPool;
use pal_async::wait::PolledWait;
use pal_event::Event;
use std::future::poll_fn;
use std::task::Poll;
use virtio::QueueResources;
use virtio::VirtioDevice;
use virtio::queue::QueueParams;
use virtio::queue::QueueState;
use virtio::spec::VirtioDeviceFeatures;
use virtio::spec::blk::VIRTIO_BLK_ID_BYTES;
use virtio::spec::blk::VIRTIO_BLK_T_DISCARD;
use virtio::spec::blk::VIRTIO_BLK_T_FLUSH;
use virtio::spec::blk::VIRTIO_BLK_T_GET_ID;
use virtio::spec::blk::VIRTIO_BLK_T_IN;
use virtio::spec::blk::VIRTIO_BLK_T_OUT;
use virtio::spec::blk::VirtioBlkDiscardWriteZeroes;
use virtio::spec::blk::VirtioBlkReqHeader;
use virtio::spec::queue::DescriptorFlags;
use virtio::test_helpers::init_avail_ring;
use virtio::test_helpers::init_used_ring;
use virtio::test_helpers::make_available;
use virtio::test_helpers::read_used;
use virtio::test_helpers::write_descriptor;
use virtio_blk::VirtioBlkDevice;
use vmcore::interrupt::Interrupt;
use vmcore::vm_task::SingleDriverBackend;
use vmcore::vm_task::VmTaskDriverSource;
use xtask_fuzz::fuzz_target;
use zerocopy::IntoBytes;

// --- Bounds ---
//
// Every bound below exists so that a single fuzz input costs a small, roughly
// constant amount of time and memory. A libFuzzer campaign is only useful if
// tens of thousands of inputs run per minute; an unbounded harness spends all
// of its time in a handful of enormous inputs instead of exploring the parser.

/// Queue size. Must be a power of two (split rings only) and non-zero. Kept
/// small so the descriptor table, available ring, and used ring all stay in a
/// few pages and so descriptor index reuse (which is itself interesting — it
/// lets a later chain rewrite the links of an earlier one) happens quickly.
const QUEUE_SIZE: u16 = 32;

/// Total guest memory. Descriptor table, rings, and request data all live here;
/// GPAs at or beyond this are out of range and must be rejected by the device.
const MEM_SIZE: u64 = 0x40000;
const DESC_ADDR: u64 = 0x0000;
const AVAIL_ADDR: u64 = 0x1000;
const USED_ADDR: u64 = 0x2000;
/// Start of the region used for request headers, payloads, and status bytes.
const DATA_BASE: u64 = 0x10000;
/// End of that region. Deliberately smaller than the aggregate payload bound
/// below, so that [`Harness::alloc`] wraps within a single input and later
/// chains genuinely alias the buffers of earlier ones.
const DATA_END: u64 = DATA_BASE + 0x4000;

/// A GPA guaranteed to be outside guest memory, used for the "bad GPA" mutation.
const BAD_GPA: u64 = MEM_SIZE + 0x1000;

/// Backing disk size. 128 512-byte sectors — big enough for multi-sector IO,
/// small enough that allocating a fresh RAM disk per input is cheap.
const DISK_SIZE: u64 = 64 * 1024;
const SECTOR_SIZE: u64 = 512;
const DISK_SECTORS: u64 = DISK_SIZE / SECTOR_SIZE;

/// Maximum payload bytes in a single descriptor chain.
const MAX_CHAIN_PAYLOAD_BYTES: u32 = 4 * 4096;

/// Maximum aggregate payload bytes across all chains in one input.
///
/// This is the bounce-buffer bound. `virtio_blk`'s `do_io` carries a
/// `// TODO: cap data_len to a reasonable maximum (e.g. seg_max * PAGE_SIZE)`
/// and, on the non-`PagedRange`-compatible path, calls
/// `GuestMemory::allocate(data_len as usize)` with a length that comes straight
/// from the guest's descriptor lengths. An unfuzzed harness can therefore ask
/// the device for an arbitrarily large host allocation per request, and the
/// campaign drowns in allocation noise (slow inputs, OOM reports) instead of
/// exploring the request parser. The natural reference point is the device's
/// own advertised limit, `DEFAULT_SEG_MAX = virtio::DEFAULT_QUEUE_SIZE - 2 =
/// 254` segments; a handful
/// of pages per input is far below anything a real driver would consider a
/// large request and still exercises both the fast path and the bounce path.
/// If the product ever grows a real cap, this bound can be raised to sit just
/// above it so the cap itself becomes reachable.
const MAX_TOTAL_PAYLOAD_BYTES: u32 = 8 * 4096;

/// Maximum descriptors in one chain, including header and status.
const MAX_DESCS_PER_CHAIN: usize = 8;

/// A chain must fit in the descriptor table, which `submit` relies on when it
/// hands out consecutive descriptor indices.
const _: () = assert!(MAX_DESCS_PER_CHAIN <= QUEUE_SIZE as usize);

/// Maximum number of payload fragments a chain's data is split across.
const MAX_PAYLOAD_FRAGS: u8 = 4;

/// Maximum number of chains published in one input. Below `QUEUE_SIZE` so the
/// harness does not trivially exhaust the ring on every input, but chains that
/// are never completed still accumulate and can hit the in-flight limit.
const MAX_CHAINS: u32 = 16;

/// Maximum number of fuzz actions per input.
const MAX_ACTIONS: u32 = 32;

/// Maximum number of executor yields per input.
///
/// Each yield is one opportunity for the device worker task to run. This is the
/// hang budget: rather than waiting for a completion that a malformed chain may
/// legitimately never produce, the harness simply runs out of yields and moves
/// on. A device that genuinely wedges shows up as a libFuzzer timeout, not as a
/// panic from this harness.
const MAX_YIELDS: u32 = 48;

/// Yields reserved for teardown, so that in-flight work has a chance to drain
/// even if the action loop burned the whole budget.
const TEARDOWN_YIELDS: u32 = 4;

/// Maximum used-ring entries drained in a single drain action.
const MAX_DRAIN_ENTRIES: u32 = 64;

// --- Grammar ---

#[derive(Debug, Arbitrary)]
enum Action {
    /// Build a descriptor chain and publish it in the available ring.
    Submit(ChainSpec),
    /// Signal the queue event, as a guest driver's notification would.
    Kick,
    /// Let the device worker task run.
    Yield,
    /// Consume completions from the used ring.
    DrainUsed,
    /// Stop queue 0, keeping its returned state for a later restart.
    StopQueue,
    /// Exercise the transport's teardown ordering: stop every queue, then call
    /// `VirtioDevice::reset`. virtio-blk does not override `reset`, so the
    /// reset call itself is currently the trait's default no-op; what this
    /// action covers is the stop-then-reset ordering and the discard of any
    /// saved queue state, not a device-specific reset path.
    Reset,
    /// Restart queue 0, either fresh or from the saved state.
    Restart { use_saved_state: bool },
}

#[derive(Debug, Arbitrary)]
enum ReqSpec {
    In {
        sector: u64,
    },
    Out {
        sector: u64,
    },
    Flush,
    GetId,
    Discard {
        sector: u64,
        num_sectors: u32,
        flags: u32,
    },
    /// A request type the device does not implement.
    Unknown {
        request_type: u32,
        sector: u64,
    },
}

/// A descriptor chain to publish, described as a well-formed request plus a set
/// of independent mutations. Keeping the mutations orthogonal lets libFuzzer
/// reach "valid request, one thing wrong" states, which is where parser bugs
/// live, without having to rediscover the whole chain layout by chance.
#[derive(Debug, Arbitrary)]
struct ChainSpec {
    req: ReqSpec,
    /// Requested payload bytes; clamped to the per-chain and aggregate bounds.
    payload_len: u16,
    /// Round the payload up to a whole number of 512-byte sectors (well-formed
    /// for IN/OUT) instead of leaving it arbitrary (rejected with IOERR).
    sector_aligned: bool,
    /// Number of descriptors the payload is split across.
    frags: u8,
    /// Leave a gap between payload fragments, so their GPAs are neither
    /// contiguous nor page-aligned. This defeats `try_build_gpn_list` and
    /// forces the device down its bounce-buffer path.
    gap_between_frags: bool,
    /// Mark the payload descriptors device-writable (correct for IN) rather
    /// than device-readable (correct for OUT).
    writable_payload: bool,
    /// Truncate the header descriptor so the header is only partly readable.
    short_header: bool,
    /// Omit the trailing status descriptor entirely.
    omit_status: bool,
    /// Make the status descriptor device-readable, so the device has nowhere
    /// to write the status byte.
    readable_status: bool,
    /// Insert a zero-length descriptor into the middle of the chain.
    zero_len_desc: bool,
    /// Point one descriptor at a GPA outside guest memory.
    bad_gpa: bool,
    /// Set the INDIRECT flag on the head descriptor. Depending on the
    /// negotiated feature word this covers either the unnegotiated rejection
    /// path or the indirect-table parser fed with arbitrary bytes.
    indirect_flag: bool,
    /// Kick the device immediately after publishing the chain.
    kick: bool,
}

/// Pick a sector number. Half the inputs get an in-range sector so the real IO
/// path is reachable; the rest get the raw value, covering out-of-range and
/// overflow-prone sectors near `u64::MAX`.
fn pick_sector(raw: u64) -> u64 {
    if raw & 1 == 0 {
        (raw >> 1) % DISK_SECTORS
    } else {
        raw
    }
}

// --- Harness ---

struct Harness {
    mem: GuestMemory,
    device: VirtioBlkDevice,
    queue_event: Event,
    interrupt_event: Event,
    /// A private event used only to force the executor to sleep, so that the
    /// pool's IO backend polls the device's queue event. See [`Harness::yield_to_device`].
    poke_event: Event,
    poke_wait: PolledWait<Event>,
    started: bool,
    /// Feature word negotiated with the device. Drawn from the fuzz input once
    /// per input and then held fixed, because the ring layout it selects must
    /// not change across stop/restart.
    features: VirtioDeviceFeatures,
    saved_state: Option<QueueState>,
    avail_idx: u16,
    used_idx: u16,
    next_desc: u16,
    next_data: u64,
    /// Remaining aggregate payload budget, in bytes.
    payload_budget: u32,
    chains_submitted: u32,
    completions: u32,
    yields_used: u32,
}

impl Harness {
    fn new(driver: &DefaultDriver, raw_features: u64) -> Self {
        let mem = GuestMemory::allocate(MEM_SIZE as usize);
        init_avail_ring(&mem, AVAIL_ADDR);
        init_used_ring(&mem, USED_ADDR);

        let disk = disklayer_ram::ram_disk(DISK_SIZE, false).unwrap();
        let driver_source = VmTaskDriverSource::new(SingleDriverBackend::new(driver.clone()));
        let device = VirtioBlkDevice::new(&driver_source, disk, false);

        // A driver may only accept bits the device offers, so mask the fuzzed
        // word with the device's own feature word. VIRTIO_F_RING_PACKED is
        // always cleared: `virtio::test_helpers` builds split rings only, and
        // the device would then parse the rings as packed.
        let features = VirtioDeviceFeatures::from_bits(
            raw_features & device.traits().device_features.into_bits(),
        )
        .with_ring_packed(false);

        let poke_event = Event::new();
        let poke_wait = PolledWait::new(driver, poke_event.clone()).unwrap();

        Self {
            mem,
            device,
            queue_event: Event::new(),
            interrupt_event: Event::new(),
            poke_event,
            poke_wait,
            started: false,
            features,
            saved_state: None,
            avail_idx: 0,
            used_idx: 0,
            next_desc: 0,
            next_data: DATA_BASE,
            payload_budget: MAX_TOTAL_PAYLOAD_BYTES,
            chains_submitted: 0,
            completions: 0,
            yields_used: 0,
        }
    }

    /// Start queue 0.
    ///
    /// With `initial_state == None` the rings are reinitialized too, matching a
    /// guest that reprograms the queue from scratch. With a saved state, the
    /// rings are left alone, matching restore/`SET_VRING_BASE`.
    async fn start(&mut self, initial_state: Option<QueueState>) {
        if self.started {
            return;
        }
        if initial_state.is_none() {
            init_avail_ring(&self.mem, AVAIL_ADDR);
            init_used_ring(&self.mem, USED_ADDR);
            self.avail_idx = 0;
            self.used_idx = 0;
        }
        let r = self
            .device
            .start_queue(
                0,
                QueueResources {
                    params: QueueParams {
                        size: QUEUE_SIZE,
                        enable: true,
                        desc_addr: DESC_ADDR,
                        avail_addr: AVAIL_ADDR,
                        used_addr: USED_ADDR,
                    },
                    notify: Interrupt::from_event(self.interrupt_event.clone()),
                    event: self.queue_event.clone(),
                    guest_memory: self.mem.clone(),
                },
                &self.features,
                initial_state,
            )
            .await;
        // The queue parameters above are fixed and valid, so a failure here
        // would be a device bug rather than a guest-triggered condition.
        r.expect("start_queue with valid parameters must succeed");
        self.started = true;
    }

    /// Give the device worker task a chance to run.
    ///
    /// The harness and the device share one single-threaded executor whose IO
    /// backend only polls file descriptors when the executor is about to sleep.
    /// A plain self-waking yield would therefore keep the executor busy and the
    /// device's queue event — an eventfd — would never be observed. Instead,
    /// arm a wait on a private event *first* (which registers interest and
    /// returns pending), then signal it. The executor sleeps, the backend polls
    /// every registered fd in one go, and both this wait and the device's queue
    /// wait are woken.
    ///
    /// This costs one poll cycle and no wall-clock time.
    async fn yield_to_device(&mut self) {
        if self.yields_used >= MAX_YIELDS {
            return;
        }
        self.yields_used += 1;
        let poke_event = &self.poke_event;
        let poke_wait = &mut self.poke_wait;
        let mut signaled = false;
        poll_fn(|cx| match poke_wait.poll_wait(cx) {
            Poll::Ready(_) => Poll::Ready(()),
            Poll::Pending => {
                if !signaled {
                    signaled = true;
                    poke_event.signal();
                }
                Poll::Pending
            }
        })
        .await;
    }

    /// Allocate `len` bytes of guest memory for request data, wrapping around
    /// when the data region is exhausted. Reuse is intentional: the region
    /// ([`DATA_BASE`]..[`DATA_END`]) is deliberately smaller than the aggregate
    /// payload budget, so a busy input wraps and later chains alias the buffers
    /// of earlier ones — including buffers still owned by the device.
    fn alloc(&mut self, len: u32) -> u64 {
        let len = len as u64;
        if self.next_data + len > DATA_END {
            self.next_data = DATA_BASE;
        }
        let gpa = self.next_data;
        self.next_data += len;
        gpa
    }

    /// Drain up to [`MAX_DRAIN_ENTRIES`] completions from the used ring.
    fn drain_used(&mut self) {
        for _ in 0..MAX_DRAIN_ENTRIES {
            let Some((_id, _len)) = read_used(&self.mem, USED_ADDR, QUEUE_SIZE, &mut self.used_idx)
            else {
                break;
            };
            self.completions += 1;
            // The device must never complete more buffers than were made
            // available to it.
            assert!(
                self.completions <= self.chains_submitted,
                "{} completions for {} published chains",
                self.completions,
                self.chains_submitted
            );
        }
    }

    /// Build and publish one descriptor chain.
    fn submit(&mut self, spec: &ChainSpec) {
        if self.chains_submitted >= MAX_CHAINS {
            return;
        }

        // Request header (plus the discard segment, which the device reads as
        // part of the same leading readable region).
        let (request_type, sector) = match spec.req {
            ReqSpec::In { sector } => (VIRTIO_BLK_T_IN, pick_sector(sector)),
            ReqSpec::Out { sector } => (VIRTIO_BLK_T_OUT, pick_sector(sector)),
            ReqSpec::Flush => (VIRTIO_BLK_T_FLUSH, 0),
            ReqSpec::GetId => (VIRTIO_BLK_T_GET_ID, 0),
            ReqSpec::Discard { .. } => (VIRTIO_BLK_T_DISCARD, 0),
            ReqSpec::Unknown {
                request_type,
                sector,
            } => (request_type, pick_sector(sector)),
        };
        let mut header = VirtioBlkReqHeader {
            request_type,
            reserved: 0,
            sector,
        }
        .as_bytes()
        .to_vec();
        if let ReqSpec::Discard {
            sector,
            num_sectors,
            flags,
        } = spec.req
        {
            header.extend_from_slice(
                VirtioBlkDiscardWriteZeroes {
                    sector: pick_sector(sector),
                    num_sectors,
                    flags,
                }
                .as_bytes(),
            );
        }
        let header_gpa = self.alloc(header.len() as u32);
        self.mem.write_at(header_gpa, &header).unwrap();

        // Payload, capped by the per-chain and aggregate bounds.
        let mut payload_len = (spec.payload_len as u32).min(MAX_CHAIN_PAYLOAD_BYTES);
        if spec.sector_aligned {
            payload_len = payload_len.next_multiple_of(SECTOR_SIZE as u32);
        }
        if matches!(spec.req, ReqSpec::GetId) {
            // GET_ID writes a fixed-size identifier; a chain that is exactly
            // long enough is the interesting well-formed case.
            payload_len = payload_len.min(VIRTIO_BLK_ID_BYTES as u32);
        }
        let payload_len = payload_len.min(self.payload_budget);
        self.payload_budget -= payload_len;

        let frags = if payload_len == 0 {
            0
        } else {
            1 + (spec.frags % MAX_PAYLOAD_FRAGS) as u32
        };

        // (gpa, len, device-writable)
        let mut descs: Vec<(u64, u32, bool)> = Vec::new();
        let header_desc_len = if spec.short_header {
            header.len() as u32 / 2
        } else {
            header.len() as u32
        };
        descs.push((header_gpa, header_desc_len, false));

        let mut remaining = payload_len;
        for i in 0..frags {
            let len = if i + 1 == frags {
                remaining
            } else {
                (payload_len / frags).min(remaining)
            };
            remaining -= len;
            let gpa = self.alloc(len);
            if len > 0 {
                // Leave the payload deterministic rather than uninitialized.
                self.mem.fill_at(gpa, 0xcc, len as usize).unwrap();
            }
            descs.push((gpa, len, spec.writable_payload));
            if spec.gap_between_frags {
                // A gap that is neither page-aligned nor zero, so the chain
                // cannot be described by a PagedRange.
                let _gap = self.alloc(100);
            }
        }

        if spec.zero_len_desc {
            descs.push((self.alloc(0), 0, spec.writable_payload));
        }

        if !spec.omit_status {
            let status_gpa = self.alloc(1);
            self.mem.write_at(status_gpa, &[0xff]).unwrap();
            descs.push((status_gpa, 1, !spec.readable_status));
        }

        descs.truncate(MAX_DESCS_PER_CHAIN);
        if spec.bad_gpa {
            let i = descs.len() - 1;
            descs[i].0 = BAD_GPA;
        }

        let n = descs.len();
        // Descriptor indices are consecutive; wrap when the table runs out.
        if self.next_desc as usize + n > QUEUE_SIZE as usize {
            self.next_desc = 0;
        }
        let head = self.next_desc;
        self.next_desc += n as u16;

        for (i, &(gpa, len, writable)) in descs.iter().enumerate() {
            let index = head + i as u16;
            let flags = DescriptorFlags::new()
                .with_next(i + 1 < n)
                .with_write(writable)
                .with_indirect(i == 0 && spec.indirect_flag);
            write_descriptor(&self.mem, DESC_ADDR, index, gpa, len, flags, index + 1);
        }

        make_available(&self.mem, AVAIL_ADDR, QUEUE_SIZE, head, &mut self.avail_idx);
        self.chains_submitted += 1;

        if spec.kick {
            self.queue_event.signal();
        }
    }

    async fn apply(&mut self, action: Action) {
        match action {
            Action::Submit(spec) => self.submit(&spec),
            Action::Kick => self.queue_event.signal(),
            Action::Yield => self.yield_to_device().await,
            Action::DrainUsed => self.drain_used(),
            Action::StopQueue => {
                if self.started {
                    self.saved_state = self.device.stop_queue(0).await;
                    self.started = false;
                }
            }
            Action::Reset => {
                // Transport order: all queues are stopped before reset.
                if self.started {
                    self.saved_state = self.device.stop_queue(0).await;
                    self.started = false;
                }
                self.device.reset().await;
                self.saved_state = None;
            }
            Action::Restart { use_saved_state } => {
                let state = if use_saved_state {
                    self.saved_state.take()
                } else {
                    None
                };
                self.start(state).await;
            }
        }
    }

    async fn run(&mut self, u: &mut Unstructured<'_>) -> arbitrary::Result<()> {
        self.start(None).await;
        let mut actions = 0;
        while actions < MAX_ACTIONS && !u.is_empty() {
            actions += 1;
            let action: Action = u.arbitrary()?;
            self.apply(action).await;
        }
        Ok(())
    }

    /// Tear the device down the way a transport does: stop every queue, then
    /// reset. Both must complete even after a protocol violation retired the
    /// queue, which is the wedged-worker regression this exercises.
    async fn teardown(&mut self) {
        // Give any work that was kicked but not yet processed a bounded chance
        // to drain, so teardown is exercised with IOs in flight as well as idle.
        self.yields_used = self.yields_used.saturating_sub(TEARDOWN_YIELDS);
        for _ in 0..TEARDOWN_YIELDS {
            self.yield_to_device().await;
        }
        self.device.stop_queue(0).await;
        self.started = false;
        self.device.reset().await;
        self.drain_used();
    }
}

async fn do_fuzz(driver: DefaultDriver, u: &mut Unstructured<'_>) -> arbitrary::Result<()> {
    let raw_features: u64 = u.arbitrary()?;
    let mut harness = Harness::new(&driver, raw_features);
    let r = harness.run(u).await;
    harness.teardown().await;
    r
}

fuzz_target!(|input: &[u8]| -> libfuzzer_sys::Corpus {
    xtask_fuzz::init_tracing_if_repro();
    let mut u = Unstructured::new(input);
    let r = DefaultPool::run_with(async |driver| do_fuzz(driver, &mut u).await);
    if r.is_err() {
        libfuzzer_sys::Corpus::Reject
    } else {
        libfuzzer_sys::Corpus::Keep
    }
});
