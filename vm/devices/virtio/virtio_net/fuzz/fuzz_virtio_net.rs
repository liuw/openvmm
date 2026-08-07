// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![cfg_attr(all(target_os = "linux", target_env = "gnu"), no_main)]

//! Fuzzes the virtio-net device with guest-controlled descriptor rings and a
//! programmable network backend.
//!
//! virtio-net sits between two mutually distrustful parties: a guest driver
//! that writes raw descriptor chains into shared memory, and a
//! [`net_backend::Endpoint`] that completes receives and transmits at times the
//! device cannot control. Nearly every interesting bug in a virtqueue frontend
//! lives in the interleaving of those two: a descriptor chain retired twice, a
//! receive buffer injected into after the queue was torn down, a transmit
//! partially consumed by the backend and re-presented on the next poll.
//!
//! So this target fuzzes *guest bytes and event ordering*, not the backend
//! contract. The mock endpoint/queue below is deliberately well-behaved — it
//! only completes buffers the device actually handed it, exactly once, on whole
//! packet boundaries — because the device is entitled to assert on a backend
//! that lies (`pending_tx_packets[id].take().unwrap()`, `expect("valid packet
//! index")`, and the whole-packet `unreachable!()` in the synchronous transmit
//! path). What the fuzzer *does* control is when those legal completions
//! happen, relative to guest kicks, queue stops, resets and ring rebuilds.
//! The oracle is "no panic, no hang".
//!
//! Everything is driven by a single-threaded executor and a bounded yield
//! budget: there are no timeouts, no sleeps and no wall-clock waits anywhere in
//! this harness, so a slow machine can never turn into a spurious fuzz failure.
//! Running out of budget is a normal outcome — the input simply stops making
//! progress.
//!
//! Liveness is load-bearing here and is easy to break silently: the guest kicks
//! are eventfds, and the device only observes them when the executor actually
//! sleeps in `epoll_wait`. See [`Harness::yield_now`] for why every yield in
//! this harness goes through a real event wait rather than a self-waking
//! `Poll::Pending`. Under a repro run (`XTASK_FUZZ_REPRO`) the harness prints
//! kick / receive-delivered / transmit-completed counts on the way out, so a
//! future change that re-breaks liveness is visible instead of silent.

use anyhow::Context as _;
use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use async_trait::async_trait;
use guestmem::GuestMemory;
use inspect::InspectMut;
use net_backend::BufferAccess;
use net_backend::Endpoint;
use net_backend::QueueConfig;
use net_backend::RssConfig;
use net_backend::RxBufferSegment;
use net_backend::RxChecksumState;
use net_backend::RxId;
use net_backend::RxMetadata;
use net_backend::TxError;
use net_backend::TxId;
use net_backend::TxOffloadSupport;
use net_backend::TxSegment;
use net_backend::TxSegmentType;
use net_backend_resources::mac_address::MacAddress;
use pal_async::DefaultDriver;
use pal_async::DefaultPool;
use pal_async::wait::PolledWait;
use pal_event::Event;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use virtio::QueueResources;
use virtio::VirtioDevice;
use virtio::queue::QueueParams;
use virtio::spec::VirtioDeviceFeatures;
use virtio::spec::queue::DescriptorFlags;
use virtio::test_helpers::init_avail_ring;
use virtio::test_helpers::init_used_ring;
use virtio::test_helpers::make_available;
use virtio::test_helpers::read_used;
use virtio::test_helpers::write_descriptor;
use virtio_net::Device;
use vmcore::interrupt::Interrupt;
use vmcore::vm_task::SingleDriverBackend;
use vmcore::vm_task::VmTaskDriverSource;
use xtask_fuzz::fuzz_target;

// ---------------------------------------------------------------------------
// Caps and bounds.
//
// Every one of these exists to keep a single fuzz input bounded in time and
// memory. Hitting any of them is a normal outcome: the corresponding action
// becomes a no-op, never a panic.
// ---------------------------------------------------------------------------

/// Virtqueue size. Small enough that the 16-entry descriptor table is easy to
/// saturate (which is what makes ring-full and duplicate-index handling
/// reachable), and a legal power of two <= 1<<15.
const QUEUE_SIZE: u16 = 16;

/// Size of the virtio-net header the device expects at the front of every
/// transmit chain. Mirrors the crate-private `virtio_net::header_size()`
/// (`offset_of!(VirtioNetHeader, hash_value)`), which is not exported. A wrong
/// guess here would only change which drop path the guest hits, never the
/// validity of the harness.
const NET_HEADER_SIZE: u32 = 12;

/// Maximum number of fuzz actions per input. Bounds total executor work.
const MAX_ACTIONS: usize = 256;

/// Total executor yields available to one input. Each yield lets the spawned
/// coordinator/worker tasks run. This is the substitute for a timeout: work
/// that has not converged within the budget is simply abandoned.
const MAX_YIELDS: u32 = 512;

/// Yields a single `Yield` action may consume.
const MAX_YIELDS_PER_ACTION: u32 = 8;

/// Yields spent waiting for `get_queues` to hand back the mock's control
/// handle after the second `start_queue`.
const HANDLE_YIELD_BUDGET: u32 = 32;

/// Maximum length of a single guest buffer (one descriptor).
const MAX_BUF_LEN: u32 = 512;

/// Maximum data segments in one transmit chain, on top of the header
/// descriptor.
const MAX_TX_DATA_SEGMENTS: usize = 4;

/// Maximum writable segments in one receive chain.
const MAX_RX_SEGMENTS: usize = 2;

/// Aggregate guest bytes an input may describe across all posted buffers.
const MAX_TOTAL_BYTES: u64 = 1 << 20;

/// Used-ring entries drained by a single `DrainUsed` action.
const MAX_DRAIN: usize = 64;

/// Backend-side queue depth caps, so a stream of inject/complete actions cannot
/// grow unbounded memory when the device is not consuming them.
const MAX_PENDING_INJECT: usize = 64;
const MAX_TX_COMPLETE_PER_ACTION: usize = 8;

/// Whole packets a single `tx_avail` may consume, as selected by `SetTxMode`.
///
/// The fuzzer's byte is mapped into `1..=MAX_TX_PACKETS_PER_AVAIL`: zero would
/// mean "consume nothing, ever", which wedges the transmit half for the rest of
/// the input (the device keeps `tx_segments` non-empty and therefore stops
/// polling the transmit kick), and a wedged device is budget spent on nothing.
const MAX_TX_PACKETS_PER_AVAIL: usize = 8;

/// One in this many `FailTxPoll` actions actually arms the failure.
///
/// A `tx_poll` error — either `TxError` variant — is a `WorkerError::Endpoint`,
/// which ends the queue worker task for that incarnation: the device is dead
/// until the harness stops and starts it again. The path is worth covering, but
/// at the raw ~12% of actions that `Arbitrary` produces it dominated the budget
/// of most inputs. Rate-limit it, and recover explicitly when it does fire.
const FAIL_TX_POLL_RATE: u8 = 8;

/// Yields spent letting the worker observe an armed `tx_poll` failure before
/// the harness restarts the device.
const FAIL_TX_POLL_YIELDS: u32 = 4;

// Guest memory layout. Rings are page-aligned and disjoint; everything above
// `DATA_BASE` is the buffer arena.
const RX_DESC_ADDR: u64 = 0x0000;
const RX_AVAIL_ADDR: u64 = 0x1000;
const RX_USED_ADDR: u64 = 0x2000;
const TX_DESC_ADDR: u64 = 0x10000;
const TX_AVAIL_ADDR: u64 = 0x11000;
const TX_USED_ADDR: u64 = 0x12000;
const DATA_BASE: u64 = 0x20000;
const TOTAL_MEM_SIZE: usize = 0x40000;

/// `VIRTIO_F_RING_PACKED` (bit 34). Never negotiated: `virtio::test_helpers`
/// only builds split rings, and mixing ring formats across a restart is a
/// documented panic in `virtio::queue`.
const RING_PACKED_BIT: u64 = 1 << 34;

// ---------------------------------------------------------------------------
// Mock network backend.
//
// A single `QueueShared` instance is created per `get_queues` call and shared
// between the `MockQueue` (owned by the device) and the harness. Fresh state
// per call is what keeps the backend contract honest across restarts: receive
// IDs and in-flight transmit IDs from a previous incarnation are dropped with
// the old state rather than being replayed into a device that has already
// forgotten them.
// ---------------------------------------------------------------------------

/// A queued request to deliver a received packet into whichever receive buffer
/// the device posts next.
struct RxInject {
    len: u32,
    ip_checksum: RxChecksumState,
    l4_checksum: RxChecksumState,
    /// Deliver via the discontiguous `write_packet_segments` path (and query
    /// the buffer's guest addresses first, as a DMA-capable backend would)
    /// rather than the single-slice `write_packet` path.
    segmented: bool,
}

struct QueueShared {
    /// Receive buffer IDs handed to us by `rx_avail` and not yet returned from
    /// `rx_poll`. Injecting into anything else would trip the device's
    /// `expect("valid packet index")`.
    rx_pending: VecDeque<RxId>,
    /// Fuzz-requested receives, consumed by `rx_poll`.
    rx_inject: VecDeque<RxInject>,
    /// Transmit IDs consumed asynchronously and not yet completed.
    tx_inflight: VecDeque<TxId>,
    /// Transmit IDs the fuzzer has released; drained by `tx_poll`.
    tx_done: VecDeque<TxId>,
    /// Whether `tx_avail` reports synchronous completion.
    tx_sync: bool,
    /// Whole packets a single `tx_avail` will consume. Never zero — see
    /// [`MAX_TX_PACKETS_PER_AVAIL`].
    tx_max_packets: usize,
    /// A one-shot error for the next `tx_poll`; `true` for fatal.
    tx_poll_error: Option<bool>,
    waker: Option<Waker>,
}

impl QueueShared {
    fn new() -> Self {
        Self {
            rx_pending: VecDeque::new(),
            rx_inject: VecDeque::new(),
            tx_inflight: VecDeque::new(),
            tx_done: VecDeque::new(),
            tx_sync: true,
            tx_max_packets: usize::MAX,
            tx_poll_error: None,
            waker: None,
        }
    }

    /// True when a subsequent `rx_poll`/`tx_poll` is guaranteed to consume at
    /// least one item.
    ///
    /// This is load-bearing: the device's poll loop re-runs the whole work
    /// cycle every time `poll_ready` returns `Ready`, so a readiness signal
    /// that does not correspond to consumable work would spin the worker task
    /// forever without ever yielding to the executor — a hang, not a failure.
    fn ready(&self) -> bool {
        self.tx_poll_error.is_some()
            || !self.tx_done.is_empty()
            || (!self.rx_inject.is_empty() && !self.rx_pending.is_empty())
    }

    fn wake(&mut self) {
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }
}

/// Liveness counters, kept across restarts and reported only on a repro run.
///
/// These exist because the failure mode they watch for is silent: a harness
/// that never wakes the device still "passes" every input, just without
/// executing any of the device it is supposed to be fuzzing.
#[derive(Default)]
struct Stats {
    kicks: AtomicUsize,
    rx_delivered: AtomicUsize,
    tx_completed: AtomicUsize,
}

impl Stats {
    /// Bumps a counter, but only under a repro run, so a normal fuzz run pays
    /// nothing beyond a `OnceLock` read.
    fn bump(counter: &AtomicUsize, n: usize) {
        if xtask_fuzz::is_repro() {
            counter.fetch_add(n, Ordering::Relaxed);
        }
    }
}

struct MockQueue {
    shared: Arc<Mutex<QueueShared>>,
    stats: Arc<Stats>,
}

impl InspectMut for MockQueue {
    fn inspect_mut(&mut self, req: inspect::Request<'_>) {
        req.ignore();
    }
}

#[async_trait]
impl net_backend::Queue for MockQueue {
    fn poll_ready(&mut self, cx: &mut Context<'_>, _pool: &mut dyn BufferAccess) -> Poll<()> {
        let mut shared = self.shared.lock();
        if shared.ready() {
            return Poll::Ready(());
        }
        shared.waker = Some(cx.waker().clone());
        Poll::Pending
    }

    fn rx_avail(&mut self, _pool: &mut dyn BufferAccess, done: &[RxId]) {
        let mut shared = self.shared.lock();
        shared.rx_pending.extend(done.iter().copied());
    }

    fn rx_poll(
        &mut self,
        pool: &mut dyn BufferAccess,
        packets: &mut [RxId],
    ) -> anyhow::Result<usize> {
        let mut shared = self.shared.lock();
        let mut n = 0;
        while n < packets.len() {
            if shared.rx_pending.is_empty() {
                // Nothing to deliver into. Leave the injects queued rather than
                // consuming them: the device runs `rx_poll` *before* `rx_avail`
                // in each work cycle, so popping here would destroy the first
                // inject of every incarnation — which is to say, in practice,
                // all of them.
                break;
            }
            let Some(inject) = shared.rx_inject.pop_front() else {
                break;
            };
            let id = shared.rx_pending.pop_front().expect("checked non-empty");
            let cap = pool.capacity(id);
            if cap == 0 {
                // The device requires `metadata.len > 0`, so a zero-capacity
                // buffer cannot be completed with data. Leave it posted.
                shared.rx_pending.push_front(id);
                continue;
            }
            let len = inject.len.clamp(1, cap);
            let data = vec![0xa5u8; len as usize];
            let metadata = RxMetadata {
                offset: 0,
                len: len as usize,
                ip_checksum: inject.ip_checksum,
                l4_checksum: inject.l4_checksum,
                ..Default::default()
            };
            if inject.segmented && len >= 2 {
                // What a backend that writes guest memory itself would do:
                // look at where the buffer actually lives, then hand over a
                // frame whose bytes are not contiguous.
                let mut segments: Vec<RxBufferSegment> = Vec::new();
                pool.push_guest_addresses(id, &mut segments);
                let (head, tail) = data.split_at(len as usize / 2);
                pool.write_packet_segments(id, &metadata, &[head, tail]);
            } else {
                pool.write_packet(id, &metadata, &data);
            }
            packets[n] = id;
            n += 1;
        }
        Stats::bump(&self.stats.rx_delivered, n);
        Ok(n)
    }

    fn tx_avail(
        &mut self,
        _pool: &mut dyn BufferAccess,
        segments: &[TxSegment],
    ) -> anyhow::Result<(bool, usize)> {
        let mut shared = self.shared.lock();
        // Consume only whole packets: the device's synchronous path walks the
        // consumed prefix packet by packet and hits `unreachable!()` if the
        // count lands mid-packet.
        let mut sent = 0;
        let mut packets = 0;
        let mut heads = Vec::new();
        while sent < segments.len() && packets < shared.tx_max_packets {
            let TxSegmentType::Head(metadata) = &segments[sent].ty else {
                break;
            };
            let count = metadata.segment_count as usize;
            if count == 0 || sent + count > segments.len() {
                break;
            }
            heads.push(metadata.id);
            sent += count;
            packets += 1;
        }
        if shared.tx_sync {
            // The device completes these itself; recording them too would
            // complete each packet twice.
            Ok((true, sent))
        } else {
            shared.tx_inflight.extend(heads);
            Ok((false, sent))
        }
    }

    fn tx_poll(
        &mut self,
        _pool: &mut dyn BufferAccess,
        done: &mut [TxId],
    ) -> Result<usize, TxError> {
        let mut shared = self.shared.lock();
        if let Some(fatal) = shared.tx_poll_error.take() {
            let err = anyhow::anyhow!("fuzzer-injected tx failure");
            return Err(if fatal {
                TxError::Fatal(err)
            } else {
                TxError::TryRestart(err)
            });
        }
        let n = done.len().min(shared.tx_done.len());
        for slot in done.iter_mut().take(n) {
            *slot = shared.tx_done.pop_front().unwrap();
        }
        Stats::bump(&self.stats.tx_completed, n);
        Ok(n)
    }
}

/// Hands out one `MockQueue` per requested queue config and publishes its
/// shared state through a non-blocking inbox.
///
/// `get_queues` is called by the coordinator task, asynchronously, some time
/// after the second `start_queue` returns, so the harness cannot receive the
/// handle inline. It polls the inbox under a yield budget instead.
struct MockEndpoint {
    inbox: Arc<Mutex<Option<Arc<Mutex<QueueShared>>>>>,
    stats: Arc<Stats>,
}

impl InspectMut for MockEndpoint {
    fn inspect_mut(&mut self, req: inspect::Request<'_>) {
        req.ignore();
    }
}

#[async_trait]
impl Endpoint for MockEndpoint {
    fn endpoint_type(&self) -> &'static str {
        "fuzz-mock"
    }

    async fn get_queues(
        &mut self,
        config: Vec<QueueConfig>,
        _rss: Option<&RssConfig<'_>>,
        queues: &mut Vec<Box<dyn net_backend::Queue>>,
    ) -> anyhow::Result<()> {
        // The device asserts that exactly `config.len()` queues come back.
        for _ in 0..config.len() {
            let shared = Arc::new(Mutex::new(QueueShared::new()));
            *self.inbox.lock() = Some(shared.clone());
            queues.push(Box::new(MockQueue {
                shared,
                stats: self.stats.clone(),
            }));
        }
        Ok(())
    }

    async fn stop(&mut self) {}

    /// virtio-net refuses to build on an unordered backend, and this mock does
    /// complete in order.
    fn is_ordered(&self) -> bool {
        true
    }

    fn tx_offload_support(&self) -> TxOffloadSupport {
        // Advertise everything so the device offers the widest feature set for
        // the fuzzer to select from.
        TxOffloadSupport {
            ipv4_header: true,
            tcp: true,
            udp: true,
            tso: true,
            uso: true,
        }
    }

    /// Single-driver layout: transmits complete on the signalling processor.
    fn tx_fast_completions(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Guest-side ring bookkeeping.
// ---------------------------------------------------------------------------

/// Tracks one split virtqueue from the guest's side: ring indices plus which
/// descriptor slots are currently owned by the device.
///
/// Slot accounting matters. Re-using a descriptor index that is still in flight
/// is a fatal `WorkerError::DuplicateDescriptor` that permanently wedges the
/// queue, and overrunning the available ring would silently replay stale
/// entries. Neither is a crash, but both end the input's ability to make
/// progress, so the harness avoids them by construction and lets the fuzzer
/// spend its budget on descriptor *contents* and event ordering instead.
struct GuestQueue {
    desc_addr: u64,
    avail_addr: u64,
    used_addr: u64,
    avail_idx: u16,
    used_idx: u16,
    /// Slot occupied by an in-flight chain.
    busy: [bool; QUEUE_SIZE as usize],
    /// For a chain head, the number of slots it occupies; 0 otherwise.
    chain_len: [u8; QUEUE_SIZE as usize],
}

impl GuestQueue {
    fn new(desc_addr: u64, avail_addr: u64, used_addr: u64) -> Self {
        Self {
            desc_addr,
            avail_addr,
            used_addr,
            avail_idx: 0,
            used_idx: 0,
            busy: [false; QUEUE_SIZE as usize],
            chain_len: [0; QUEUE_SIZE as usize],
        }
    }

    /// Re-initializes the rings in guest memory and forgets all in-flight
    /// descriptors. Used on rebuild and restart, where the device also starts
    /// from index zero.
    fn reset(&mut self, mem: &GuestMemory) {
        init_avail_ring(mem, self.avail_addr);
        init_used_ring(mem, self.used_addr);
        self.avail_idx = 0;
        self.used_idx = 0;
        self.busy = [false; QUEUE_SIZE as usize];
        self.chain_len = [0; QUEUE_SIZE as usize];
    }

    /// Reserves `n` contiguous free descriptor slots, or `None` if the table is
    /// too full.
    fn alloc(&mut self, n: usize) -> Option<u16> {
        if n == 0 || n > QUEUE_SIZE as usize {
            return None;
        }
        let head =
            (0..=QUEUE_SIZE as usize - n).find(|&i| self.busy[i..i + n].iter().all(|b| !b))?;
        for slot in &mut self.busy[head..head + n] {
            *slot = true;
        }
        self.chain_len[head] = n as u8;
        Some(head as u16)
    }

    /// Releases the slots owned by a chain once its head appears in the used
    /// ring.
    fn free(&mut self, head: u16) {
        let head = head as usize;
        if head >= QUEUE_SIZE as usize {
            return;
        }
        let n = std::mem::take(&mut self.chain_len[head]) as usize;
        for slot in self.busy.iter_mut().skip(head).take(n) {
            *slot = false;
        }
    }

    fn make_available(&mut self, mem: &GuestMemory, head: u16) {
        make_available(mem, self.avail_addr, QUEUE_SIZE, head, &mut self.avail_idx);
    }

    fn drain_used(&mut self, mem: &GuestMemory) {
        for _ in 0..MAX_DRAIN {
            let Some((id, _len)) = read_used(mem, self.used_addr, QUEUE_SIZE, &mut self.used_idx)
            else {
                break;
            };
            self.free(id);
        }
    }
}

/// Bump allocator over the guest buffer arena, with a global byte budget.
struct DataAlloc {
    next: u64,
    budget: u64,
}

impl DataAlloc {
    fn new() -> Self {
        Self {
            next: DATA_BASE,
            budget: MAX_TOTAL_BYTES,
        }
    }

    /// Returns a guest address for `len` bytes, wrapping within the arena.
    ///
    /// Wrapping can alias a still-live buffer. That is harmless — guest memory
    /// contents are pure data to the device — and is itself worth exercising.
    fn alloc(&mut self, len: u32) -> Option<u64> {
        let len = len as u64;
        if len > self.budget {
            return None;
        }
        self.budget -= len;
        if self.next + len > TOTAL_MEM_SIZE as u64 {
            self.next = DATA_BASE;
        }
        let gpa = self.next;
        self.next += len;
        Some(gpa)
    }
}

// ---------------------------------------------------------------------------
// Fuzz actions.
// ---------------------------------------------------------------------------

#[derive(Debug, Arbitrary)]
enum Action {
    /// Post a writable receive chain and kick the RX queue.
    PostRx { segments: u8, len: u16 },
    /// Post a transmit chain: a header descriptor whose bytes come straight
    /// from the fuzzer, followed by data descriptors.
    PostTx {
        header: [u8; NET_HEADER_SIZE as usize],
        header_len: u8,
        segments: u8,
        len: u16,
    },
    /// Signal a queue's kick event without posting anything.
    Kick { rx: bool },
    /// Ask the backend to deliver a received packet.
    InjectRx {
        len: u16,
        ip: u8,
        l4: u8,
        segmented: bool,
    },
    /// Choose synchronous vs asynchronous transmit completion, and how many
    /// whole packets a single `tx_avail` consumes.
    SetTxMode { sync: bool, max_packets: u8 },
    /// Release asynchronously-consumed transmits for the next `tx_poll`.
    CompleteTx { count: u8 },
    /// Make the next `tx_poll` fail, then recover the device.
    ///
    /// `key` rate-limits the action (see [`FAIL_TX_POLL_RATE`]) and `features`
    /// is used for the restart that follows.
    FailTxPoll { fatal: bool, key: u8, features: u64 },
    /// Let the device's tasks run.
    Yield { count: u8 },
    /// Drain the used ring, freeing descriptor slots.
    DrainUsed { rx: bool },
    /// Stop both queues.
    Stop,
    /// Stop both queues and reset the device.
    Reset,
    /// Re-initialize the rings in guest memory and forget in-flight
    /// descriptors (only meaningful while stopped).
    RebuildRings,
    /// Start the queues, if stopped, with a fresh feature set.
    Start { features: u64 },
    /// Stop, reset, rebuild the rings and start again.
    Restart { features: u64 },
}

fn checksum_state(v: u8) -> RxChecksumState {
    match v % 4 {
        0 => RxChecksumState::Unknown,
        1 => RxChecksumState::Good,
        2 => RxChecksumState::Bad,
        _ => RxChecksumState::ValidatedButWrong,
    }
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------

struct Harness {
    device: Device,
    mem: GuestMemory,
    rx: GuestQueue,
    tx: GuestQueue,
    rx_event: Event,
    tx_event: Event,
    rx_interrupt: Interrupt,
    tx_interrupt: Interrupt,
    data: DataAlloc,
    inbox: Arc<Mutex<Option<Arc<Mutex<QueueShared>>>>>,
    shared: Option<Arc<Mutex<QueueShared>>>,
    stats: Arc<Stats>,
    /// Private event used to drive one executor sleep per yield; see
    /// [`Harness::yield_now`].
    poke_event: Event,
    poke_wait: PolledWait<Event>,
    /// Feature bits the device actually offers, minus packed ring.
    allowed_features: u64,
    enabled: bool,
    yields_left: u32,
    /// Transmit completion mode, held here rather than in the per-queue state
    /// so that it survives a restart: otherwise every restart would silently
    /// revert to synchronous completion and the asynchronous path would be
    /// exercised far less than the fuzzer asked for.
    tx_sync: bool,
    tx_max_packets: usize,
}

impl Harness {
    fn new(driver_source: &VmTaskDriverSource, driver: &DefaultDriver) -> anyhow::Result<Self> {
        let mem = GuestMemory::allocate(TOTAL_MEM_SIZE);
        let inbox = Arc::new(Mutex::new(None));
        let stats = Arc::new(Stats::default());
        let endpoint = MockEndpoint {
            inbox: inbox.clone(),
            stats: stats.clone(),
        };
        let device = Device::builder()
            .build(
                driver_source,
                Box::new(endpoint),
                MacAddress::new([0x00, 0x15, 0x5d, 0x01, 0x02, 0x03]),
            )
            .context("failed to build device")?;

        let allowed_features = device.traits().device_features.into_bits() & !RING_PACKED_BIT;

        let rx_event = Event::new();
        let tx_event = Event::new();
        let rx_interrupt = Interrupt::from_event(Event::new());
        let tx_interrupt = Interrupt::from_event(Event::new());
        let poke_event = Event::new();
        let poke_wait = PolledWait::new(driver, poke_event.clone())
            .context("failed to create the yield event wait")?;

        let mut this = Self {
            device,
            mem,
            rx: GuestQueue::new(RX_DESC_ADDR, RX_AVAIL_ADDR, RX_USED_ADDR),
            tx: GuestQueue::new(TX_DESC_ADDR, TX_AVAIL_ADDR, TX_USED_ADDR),
            rx_event,
            tx_event,
            rx_interrupt,
            tx_interrupt,
            data: DataAlloc::new(),
            inbox,
            shared: None,
            stats,
            poke_event,
            poke_wait,
            allowed_features,
            enabled: false,
            yields_left: MAX_YIELDS,
            tx_sync: true,
            tx_max_packets: usize::MAX,
        };
        this.rebuild_rings();
        Ok(this)
    }

    fn rebuild_rings(&mut self) {
        self.rx.reset(&self.mem);
        self.tx.reset(&self.mem);
    }

    /// Yields to the executor once, if the budget allows. Returns false when
    /// exhausted.
    ///
    /// This must be a *real* executor sleep, not a self-waking
    /// `cx.waker().wake_by_ref(); Poll::Pending`. The harness and the device
    /// share one single-threaded `DefaultPool`, whose Linux backend only enters
    /// `epoll_wait` — and therefore only ever observes a file descriptor — when
    /// the run queue is quiescent (`can_sleep()` in
    /// `pal_async/src/unix/epoll.rs`). A self-waking yield keeps the pool
    /// permanently "run again", so the eventfds behind `rx_event`, `tx_event`
    /// and the backend's own wakeups are never polled and every guest kick in
    /// this harness is inert.
    ///
    /// Instead, arm a wait on a private event *first* — which registers epoll
    /// interest and returns pending — and only then signal it. The pool has
    /// nothing left to run, sleeps, polls every registered fd in one go, and
    /// wakes both this wait and the device's queue waits. Costs one poll cycle
    /// and no wall-clock time.
    async fn yield_now(&mut self) -> bool {
        if self.yields_left == 0 {
            return false;
        }
        self.yields_left -= 1;
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
        true
    }

    async fn start(&mut self, features: u64) {
        if self.enabled {
            return;
        }
        // A started queue always begins at available index zero, so the guest
        // must re-initialize the rings — that is what a real driver does after
        // a device reset. Leaving stale available entries in place instead just
        // replays them into the fresh queue, which the device correctly rejects
        // as `DuplicateDescriptor`; that path is real but fatal to the queue,
        // and letting it happen by default would spend most of the fuzzer's
        // budget on a permanently wedged device.
        self.rebuild_rings();
        // Only ever negotiate bits the device offers, and never the packed
        // ring: these helpers build split rings exclusively.
        let features = VirtioDeviceFeatures::from_bits(features & self.allowed_features);
        let params = |desc_addr, avail_addr, used_addr| QueueParams {
            size: QUEUE_SIZE,
            enable: true,
            desc_addr,
            avail_addr,
            used_addr,
        };
        // Queue 0 is RX, queue 1 is TX. Both indices are constants: the device
        // only has one queue pair, and the transport is responsible for range
        // checking, so feeding fuzzed indices here would test nothing.
        self.device
            .start_queue(
                0,
                QueueResources {
                    params: params(RX_DESC_ADDR, RX_AVAIL_ADDR, RX_USED_ADDR),
                    notify: self.rx_interrupt.clone(),
                    event: self.rx_event.clone(),
                    guest_memory: self.mem.clone(),
                },
                &features,
                None,
            )
            .await
            .expect("rx start_queue with valid parameters must succeed");
        self.device
            .start_queue(
                1,
                QueueResources {
                    params: params(TX_DESC_ADDR, TX_AVAIL_ADDR, TX_USED_ADDR),
                    notify: self.tx_interrupt.clone(),
                    event: self.tx_event.clone(),
                    guest_memory: self.mem.clone(),
                },
                &features,
                None,
            )
            .await
            .expect("tx start_queue with valid parameters must succeed");
        self.enabled = true;

        // The coordinator calls `get_queues` on its own task; poll the inbox
        // under a bounded yield budget rather than waiting on a timeout.
        for _ in 0..HANDLE_YIELD_BUDGET {
            if let Some(shared) = self.inbox.lock().take() {
                {
                    let mut shared = shared.lock();
                    shared.tx_sync = self.tx_sync;
                    shared.tx_max_packets = self.tx_max_packets;
                }
                self.shared = Some(shared);
                break;
            }
            if !self.yield_now().await {
                break;
            }
        }
    }

    async fn stop(&mut self) {
        if !self.enabled {
            return;
        }
        // Teardown order matches the device's own tests: the TX half tears down
        // the active pair, the RX half is then already empty.
        self.device.stop_queue(1).await;
        self.device.stop_queue(0).await;
        self.enabled = false;
        // The device dropped the queue, so its state must not be touched again.
        self.shared = None;
        *self.inbox.lock() = None;
    }

    async fn reset(&mut self) {
        self.stop().await;
        self.device.reset().await;
    }

    fn post_rx(&mut self, segments: u8, len: u16) {
        let n = 1 + segments as usize % MAX_RX_SEGMENTS;
        let Some(head) = self.rx.alloc(n) else {
            return;
        };
        let len = len as u32 % (MAX_BUF_LEN + 1);
        for i in 0..n {
            let Some(gpa) = self.data.alloc(len) else {
                return;
            };
            let last = i == n - 1;
            let idx = head + i as u16;
            write_descriptor(
                &self.mem,
                self.rx.desc_addr,
                idx,
                gpa,
                len,
                DescriptorFlags::new().with_write(true).with_next(!last),
                if last { 0 } else { idx + 1 },
            );
        }
        self.rx.make_available(&self.mem, head);
        self.kick(true);
    }

    /// Signals a queue's kick event, as a guest driver's notification would.
    fn kick(&self, rx: bool) {
        if rx {
            self.rx_event.signal();
        } else {
            self.tx_event.signal();
        }
        Stats::bump(&self.stats.kicks, 1);
    }

    fn post_tx(&mut self, header: &[u8], header_len: u8, segments: u8, len: u16) {
        let n = 1 + segments as usize % MAX_TX_DATA_SEGMENTS;
        let Some(head) = self.tx.alloc(n + 1) else {
            return;
        };
        // Usually describe exactly the header, but sometimes a short or long
        // one so the device's header parsing sees truncated chains too.
        let header_len = match header_len % 8 {
            0 => 0,
            1 => NET_HEADER_SIZE / 2,
            2 => NET_HEADER_SIZE + 4,
            _ => NET_HEADER_SIZE,
        };
        let Some(header_gpa) = self.data.alloc(header_len.max(NET_HEADER_SIZE)) else {
            return;
        };
        // Guest-controlled header bytes: flags, gso type, csum offsets and
        // header lengths all flow into the device's offload parsing.
        self.mem.write_at(header_gpa, header).unwrap();
        write_descriptor(
            &self.mem,
            self.tx.desc_addr,
            head,
            header_gpa,
            header_len,
            DescriptorFlags::new().with_next(true),
            head + 1,
        );
        let len = len as u32 % (MAX_BUF_LEN + 1);
        for i in 0..n {
            let Some(gpa) = self.data.alloc(len) else {
                return;
            };
            let last = i == n - 1;
            let idx = head + 1 + i as u16;
            write_descriptor(
                &self.mem,
                self.tx.desc_addr,
                idx,
                gpa,
                len,
                DescriptorFlags::new().with_next(!last),
                if last { 0 } else { idx + 1 },
            );
        }
        self.tx.make_available(&self.mem, head);
        self.kick(false);
    }

    /// Runs after every action: recycle completed descriptors and give the
    /// device's tasks a chance to run.
    ///
    /// A real driver polls its used rings constantly, and without a steady
    /// supply of yields the device never observes the kicks at all. Both are
    /// what keep an input making forward progress instead of immediately
    /// filling the 16-entry descriptor table and stalling.
    async fn after_action(&mut self) {
        self.rx.drain_used(&self.mem);
        self.tx.drain_used(&self.mem);
        self.yield_now().await;
    }

    fn with_shared(&self, f: impl FnOnce(&mut QueueShared)) {
        if let Some(shared) = &self.shared {
            let mut shared = shared.lock();
            f(&mut shared);
            shared.wake();
        }
    }

    async fn run_action(&mut self, action: Action) {
        match action {
            Action::PostRx { segments, len } => self.post_rx(segments, len),
            Action::PostTx {
                header,
                header_len,
                segments,
                len,
            } => self.post_tx(&header, header_len, segments, len),
            Action::Kick { rx } => self.kick(rx),
            Action::InjectRx {
                len,
                ip,
                l4,
                segmented,
            } => self.with_shared(|shared| {
                if shared.rx_inject.len() < MAX_PENDING_INJECT {
                    shared.rx_inject.push_back(RxInject {
                        len: len as u32,
                        ip_checksum: checksum_state(ip),
                        l4_checksum: checksum_state(l4),
                        segmented,
                    });
                }
            }),
            Action::SetTxMode { sync, max_packets } => {
                // Never zero: see `MAX_TX_PACKETS_PER_AVAIL`.
                let max_packets = 1 + max_packets as usize % MAX_TX_PACKETS_PER_AVAIL;
                self.tx_sync = sync;
                self.tx_max_packets = max_packets;
                self.with_shared(|shared| {
                    shared.tx_sync = sync;
                    shared.tx_max_packets = max_packets;
                });
            }
            Action::CompleteTx { count } => self.with_shared(|shared| {
                let n = (count as usize % (MAX_TX_COMPLETE_PER_ACTION + 1))
                    .min(shared.tx_inflight.len());
                for _ in 0..n {
                    let id = shared.tx_inflight.pop_front().unwrap();
                    shared.tx_done.push_back(id);
                }
            }),
            Action::FailTxPoll {
                fatal,
                key,
                features,
            } => {
                if key % FAIL_TX_POLL_RATE == 0 {
                    self.with_shared(|shared| {
                        shared.tx_poll_error = Some(fatal);
                    });
                    // Give the worker a chance to observe the failure — that is
                    // the coverage this action is here for — and then bring the
                    // device back, since the worker task has now exited and
                    // every later action would otherwise be a no-op.
                    for _ in 0..FAIL_TX_POLL_YIELDS {
                        if !self.yield_now().await {
                            break;
                        }
                    }
                    self.reset().await;
                    self.start(features).await;
                }
            }
            Action::Yield { count } => {
                for _ in 0..(count as u32 % MAX_YIELDS_PER_ACTION + 1) {
                    if !self.yield_now().await {
                        break;
                    }
                }
            }
            Action::DrainUsed { rx } => {
                if rx {
                    self.rx.drain_used(&self.mem);
                } else {
                    self.tx.drain_used(&self.mem);
                }
            }
            Action::Stop => self.stop().await,
            Action::Reset => self.reset().await,
            Action::RebuildRings => {
                // Rebuilding under a running device would replay stale
                // available entries; the device only re-reads indices from zero
                // after a stop.
                if !self.enabled {
                    self.rebuild_rings();
                }
            }
            Action::Start { features } => self.start(features).await,
            Action::Restart { features } => {
                self.reset().await;
                self.rebuild_rings();
                self.data = DataAlloc::new();
                self.start(features).await;
            }
        }
    }
}

async fn fuzz(
    driver_source: &VmTaskDriverSource,
    driver: &DefaultDriver,
    u: &mut Unstructured<'_>,
) -> arbitrary::Result<()> {
    let mut harness = match Harness::new(driver_source, driver) {
        Ok(harness) => harness,
        Err(_) => return Ok(()),
    };
    // Pick the initial transmit completion mode from the input rather than
    // defaulting to synchronous, so that short inputs — which may never emit a
    // `SetTxMode` action — still split evenly between the two completion paths.
    harness.tx_sync = u.arbitrary()?;
    harness.start(u.arbitrary()?).await;

    for _ in 0..MAX_ACTIONS {
        if u.is_empty() {
            break;
        }
        let action = u.arbitrary()?;
        harness.run_action(action).await;
        harness.after_action().await;
    }

    // Tear down cleanly so the device's tasks are joined before the executor
    // shuts down; a stop that hangs here would be as much of a bug as a panic.
    harness.reset().await;
    xtask_fuzz::fuzz_eprintln!(
        "liveness: {} kicks, {} packets received, {} transmits completed",
        harness.stats.kicks.load(Ordering::Relaxed),
        harness.stats.rx_delivered.load(Ordering::Relaxed),
        harness.stats.tx_completed.load(Ordering::Relaxed),
    );
    Ok(())
}

fn do_fuzz(u: &mut Unstructured<'_>) -> arbitrary::Result<()> {
    DefaultPool::run_with(async |driver| {
        let driver_source = VmTaskDriverSource::new(SingleDriverBackend::new(driver.clone()));
        fuzz(&driver_source, &driver, u).await
    })
}

fuzz_target!(|input: &[u8]| -> libfuzzer_sys::Corpus {
    xtask_fuzz::init_tracing_if_repro();
    if do_fuzz(&mut Unstructured::new(input)).is_err() {
        libfuzzer_sys::Corpus::Reject
    } else {
        libfuzzer_sys::Corpus::Keep
    }
});
