#include "Data/TraceData.h"
#include "Profiler/Graph.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace proton {

class TraceData::Trace {
public:
  struct TraceContext : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    TraceContext() = default;
    explicit TraceContext(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TraceContext(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    virtual ~TraceContext() = default;

    void addChild(const Context &context, size_t id) { children[context] = id; }

    bool hasChild(const Context &context) const {
      return children.find(context) != children.end();
    }

    size_t getChild(const Context &context) const {
      return children.at(context);
    }

    size_t getParent() const { return parentId; }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::map<Context, size_t> children = {};
    friend class Trace;
  };

  struct TraceEvent {
    TraceEvent() = default;
    TraceEvent(size_t id, size_t contextId, size_t parentEventId)
        : id(id), contextId(contextId), parentEventId(parentEventId) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    size_t parentEventId = DummyId;
    size_t launchScopeEventId = DummyId;
    uint64_t cpuStartTimeNs = 0;
    uint64_t cpuEndTimeNs = 0;
    uint64_t threadId = 0;
    bool hasCpuStartTime = false;
    bool hasCpuEndTime = false;
    // Direct and linked metrics emitted for this trace event.
    DataEntry::MetricSet metricSet{};

    bool hasCpuTimeRange() const {
      return hasCpuStartTime && hasCpuEndTime && cpuEndTimeNs >= cpuStartTimeNs;
    }

    const static inline size_t DummyId = std::numeric_limits<size_t>::max();
  };

  Trace() {
    traceContextMap.try_emplace(TraceContext::RootId, TraceContext::RootId,
                                "ROOT");
  }

  size_t addContext(const Context &context, size_t parentId) {
    if (traceContextMap[parentId].hasChild(context)) {
      return traceContextMap[parentId].getChild(context);
    }
    auto id = nextTreeContextId++;
    traceContextMap.try_emplace(id, id, parentId, context.name);
    traceContextMap[parentId].addChild(context, id);
    return id;
  }

  size_t addContexts(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addContext(context, parentId);
    }
    return parentId;
  }

  size_t addContexts(const std::vector<Context> &indices) {
    auto parentId = TraceContext::RootId;
    for (auto index : indices) {
      parentId = addContext(index, parentId);
    }
    return parentId;
  }

  std::vector<Context> getContexts(size_t contextId) {
    std::vector<Context> contexts;
    auto it = traceContextMap.find(contextId);
    if (it == traceContextMap.end()) {
      throw std::runtime_error("Context not found");
    }
    std::reference_wrapper<TraceContext> context = it->second;
    contexts.push_back(context.get());
    while (context.get().parentId != TraceContext::DummyId) {
      context = traceContextMap[context.get().parentId];
      contexts.push_back(context.get());
    }
    std::reverse(contexts.begin(), contexts.end());
    return contexts;
  }

  size_t addEvent(size_t contextId,
                  size_t parentEventId = TraceEvent::DummyId) {
    traceEvents.try_emplace(nextEventId, nextEventId, contextId, parentEventId);
    return nextEventId++;
  }

  bool hasEvent(size_t eventId) {
    return traceEvents.find(eventId) != traceEvents.end();
  }

  TraceEvent &getEvent(size_t eventId) {
    auto it = traceEvents.find(eventId);
    if (it == traceEvents.end()) {
      throw std::runtime_error("Event not found");
    }
    return it->second;
  }

  void removeEvent(size_t eventId) { traceEvents.erase(eventId); }

  const std::map<size_t, TraceEvent> &getEvents() const { return traceEvents; }

private:
  size_t nextTreeContextId = TraceContext::RootId + 1;
  size_t nextEventId = 0;
  std::map<size_t, TraceEvent> traceEvents;
  // tree node id -> trace context
  std::map<size_t, TraceContext> traceContextMap;
};

thread_local std::unordered_map<const TraceData *, std::vector<size_t>>
    traceDataToActiveEventStack;

uint64_t getCurrentCpuTimestampNs() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

void TraceData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto &activeEventStack = traceDataToActiveEventStack[this];
  size_t parentEventId = activeEventStack.empty() ? Trace::TraceEvent::DummyId
                                                  : activeEventStack.back();
  auto eventId =
      currentTrace->addEvent(currentTrace->addContexts(contexts), parentEventId);
  auto &event = currentTrace->getEvent(eventId);
  event.scopeId = scope.scopeId;
  event.launchScopeEventId = eventId;
  event.cpuStartTimeNs = getCurrentCpuTimestampNs();
  event.hasCpuStartTime = true;
  event.threadId = getCurrentThreadTraceId();
  scopeIdToEventId[scope.scopeId] = eventId;
  activeEventStack.push_back(eventId);
}

void TraceData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeEventIt = scopeIdToEventId.find(scope.scopeId);
  if (scopeEventIt != scopeIdToEventId.end()) {
    auto *currentTrace = currentPhasePtrAs<Trace>();
    auto &event = currentTrace->getEvent(scopeEventIt->second);
    event.cpuEndTimeNs = getCurrentCpuTimestampNs();
    event.hasCpuEndTime = true;
    auto &activeEventStack = traceDataToActiveEventStack[this];
    if (!activeEventStack.empty() &&
        activeEventStack.back() == scopeEventIt->second) {
      activeEventStack.pop_back();
      if (activeEventStack.empty()) {
        traceDataToActiveEventStack.erase(this);
      }
    }
  }
  scopeIdToEventId.erase(scope.scopeId);
}

DataEntry TraceData::addOp(size_t phase, size_t eventId,
                           const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  auto *trace = phasePtrAs<Trace>(phase);
  auto parentContextId = 0;
  if (eventId == Data::kRootEntryId) {
    parentContextId = Trace::TraceContext::RootId;
  } else {
    auto &event = trace->getEvent(eventId);
    parentContextId = event.contextId;
  }
  const auto contextId = trace->addContexts(contexts, parentContextId);
  const auto newEventId = trace->addEvent(contextId);
  auto &newEvent = trace->getEvent(newEventId);
  if (eventId == Data::kRootEntryId) {
    newEvent.cpuStartTimeNs = getCurrentCpuTimestampNs();
    newEvent.hasCpuStartTime = true;
    newEvent.threadId = getCurrentThreadTraceId();
    auto activeEventStackIt = traceDataToActiveEventStack.find(this);
    if (activeEventStackIt != traceDataToActiveEventStack.end() &&
        !activeEventStackIt->second.empty()) {
      newEvent.launchScopeEventId = activeEventStackIt->second.back();
    }
  } else {
    const auto &parentEvent = trace->getEvent(eventId);
    if (parentEvent.scopeId != Scope::DummyScopeId) {
      newEvent.launchScopeEventId = eventId;
    } else {
      newEvent.launchScopeEventId = parentEvent.launchScopeEventId;
    }
    newEvent.cpuStartTimeNs = parentEvent.cpuStartTimeNs;
    newEvent.hasCpuStartTime = parentEvent.hasCpuStartTime;
    newEvent.threadId = parentEvent.threadId;
  }
  return DataEntry(newEventId, phase, newEvent.metricSet);
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  auto eventId = scopeIdToEventId.at(scopeId);
  auto &event = currentTrace->getEvent(eventId);
  auto &flexibleMetrics = event.metricSet.flexibleMetrics;
  for (auto [metricName, metricValue] : metrics) {
    if (flexibleMetrics.find(metricName) == flexibleMetrics.end()) {
      flexibleMetrics.emplace(metricName,
                              FlexibleMetric(metricName, metricValue));
    } else {
      flexibleMetrics.at(metricName).updateValue(metricValue);
    }
  }
}

uint64_t TraceData::getCurrentThreadTraceId() {
  auto threadId = std::this_thread::get_id();
  auto it = threadIdToTraceId.find(threadId);
  if (it != threadIdToTraceId.end()) {
    return it->second;
  }
  auto traceThreadId = nextThreadTraceId++;
  threadIdToTraceId.emplace(threadId, traceThreadId);
  return traceThreadId;
}

std::string TraceData::toJsonString(size_t phase) const {
  std::ostringstream os;
  dumpChromeTrace(os, phase);
  return os.str();
}

std::vector<uint8_t> TraceData::toMsgPack(size_t phase) const {
  std::ostringstream os;
  dumpChromeTrace(os, phase);
  MsgPackWriter writer;
  writer.packStr(os.str());
  return std::move(writer).take();
}

namespace {

// Structure to pair CycleMetric with its context for processing
struct CycleMetricWithContext {
  const CycleMetric *cycleMetric;
  // Full call path captured for this cycle metric event.
  std::vector<Context> contexts;

  CycleMetricWithContext(const CycleMetric *metric, std::vector<Context> ctx)
      : cycleMetric(metric), contexts(std::move(ctx)) {}
};

constexpr const char *kCpuThreadTidPrefix = "cpu thread ";
constexpr const char *kLaunchFlowName = "launch->kernel";
constexpr const char *kLaunchFlowCategory = "flow";
constexpr const char *kComputeMetadataScopeName = "__proton_launch_metadata";

struct OrderedTraceEvent {
  enum class Kind { Kernel, CpuScope, GraphScope };

  Kind kind;
  const KernelMetric *kernelMetric{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
  std::shared_ptr<DataEntry::FlexibleMetricMap> flexibleMetricsStorage;
  std::vector<Context> contexts;
  size_t owningEventId = TraceData::Trace::TraceEvent::DummyId;
  size_t launchScopeEventId = TraceData::Trace::TraceEvent::DummyId;
  uint64_t threadId = 0;
  size_t streamId = 0;
  uint64_t startTimeNs = 0;
  uint64_t endTimeNs = 0;
  bool isGraphLinked = false;
  bool isMetricKernel = false;
  size_t graphScopeStartIndex = 0;
  bool hasGraphFlowSource = false;
  uint64_t graphFlowStartTimeNs = 0;
  uint64_t graphFlowEndTimeNs = 0;
  std::optional<int64_t> explicitClockOffsetNs;

  const DataEntry::FlexibleMetricMap *getFlexibleMetrics() const {
    return flexibleMetricsStorage ? flexibleMetricsStorage.get()
                                  : flexibleMetrics;
  }

  static OrderedTraceEvent kernel(const KernelMetric *metric,
                                  const DataEntry::FlexibleMetricMap *metrics,
                                  std::vector<Context> contexts,
                                  size_t owningEventId, size_t streamId,
                                  size_t launchScopeEventId,
                                  bool isGraphLinked, bool isMetricKernel,
                                  size_t graphScopeStartIndex) {
    OrderedTraceEvent event;
    event.kind = Kind::Kernel;
    event.kernelMetric = metric;
    event.flexibleMetrics = metrics;
    event.contexts = std::move(contexts);
    event.owningEventId = owningEventId;
    event.launchScopeEventId = launchScopeEventId;
    event.streamId = streamId;
    event.startTimeNs =
        std::get<uint64_t>(metric->getValue(KernelMetric::StartTime));
    event.endTimeNs =
        std::get<uint64_t>(metric->getValue(KernelMetric::EndTime));
    event.isGraphLinked = isGraphLinked;
    event.isMetricKernel = isMetricKernel;
    event.graphScopeStartIndex = graphScopeStartIndex;
    return event;
  }

  static OrderedTraceEvent cpuScope(
      const DataEntry::FlexibleMetricMap *metrics, std::vector<Context> contexts,
      uint64_t threadId, uint64_t startTimeNs, uint64_t endTimeNs) {
    OrderedTraceEvent event;
    event.kind = Kind::CpuScope;
    event.flexibleMetrics = metrics;
    event.contexts = std::move(contexts);
    event.threadId = threadId;
    event.startTimeNs = startTimeNs;
    event.endTimeNs = endTimeNs;
    return event;
  }

  static OrderedTraceEvent graphScope(
      std::vector<Context> contexts, size_t streamId, uint64_t startTimeNs,
      uint64_t endTimeNs,
      std::shared_ptr<DataEntry::FlexibleMetricMap> flexibleMetricsStorage =
          nullptr,
      std::optional<int64_t> explicitClockOffsetNs = std::nullopt) {
    OrderedTraceEvent event;
    event.kind = Kind::GraphScope;
    event.contexts = std::move(contexts);
    event.streamId = streamId;
    event.startTimeNs = startTimeNs;
    event.endTimeNs = endTimeNs;
    event.flexibleMetricsStorage = std::move(flexibleMetricsStorage);
    event.explicitClockOffsetNs = explicitClockOffsetNs;
    event.flexibleMetrics = event.flexibleMetricsStorage
                                ? event.flexibleMetricsStorage.get()
                                : nullptr;
    return event;
  }
};

struct GraphKernelRecord {
  size_t orderedEventIndex{};
  size_t owningEventId{};
  size_t streamId{};
  uint64_t startTimeNs{};
  uint64_t endTimeNs{};
  size_t scopeStartIndex{};
  std::vector<Context> scopeContexts;
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
};

struct GraphScopeRange {
  std::vector<Context> contexts;
  size_t streamId{};
  uint64_t startTimeNs{};
  uint64_t endTimeNs{};
  DataEntry::FlexibleMetricMap flexibleMetrics{};
};

struct OpenGraphScope {
  std::vector<Context> contexts;
  uint64_t startTimeNs{};
};

struct GraphScopeBoundary {
  const OrderedTraceEvent *event{};
  uint64_t alignedTimeNs{};
  bool isBegin{};
  size_t insertionOrder{};
};

std::string formatFlexibleMetricValue(const MetricValueType &value) {
  return std::visit(
      [](auto &&v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, double>) {
          return std::to_string(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, std::vector<uint64_t>> ||
                             std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<double>>) {
          std::ostringstream ss;
          ss << "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) {
              ss << ",";
            }
            ss << v[i];
          }
          ss << "]";
          return ss.str();
        } else {
          static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
        }
      },
      value);
}

bool hasFlexibleMetrics(const OrderedTraceEvent &event) {
  const auto *flexibleMetrics = event.getFlexibleMetrics();
  return flexibleMetrics != nullptr && !flexibleMetrics->empty();
}

json buildCallStackJson(const std::vector<Context> &contexts) {
  json callStack = json::array();
  for (const auto &ctx : contexts) {
    callStack.push_back(ctx.name);
  }
  return callStack;
}

json buildFlexibleMetricsJson(const DataEntry::FlexibleMetricMap &flexibleMetrics) {
  json metrics = json::object();
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    metrics[metricName] = formatFlexibleMetricValue(metricValue.getValues()[0]);
  }
  return metrics;
}

std::string buildFlexibleMetricEventName(
    const std::vector<Context> &contexts,
    const DataEntry::FlexibleMetricMap &flexibleMetrics) {
  if (flexibleMetrics.empty()) {
    return GraphState::metricTag;
  }
  const auto &scopeName =
      contexts.empty() ? GraphState::metricTag : contexts.back().name;
  auto appendTuple =
      [&](std::ostringstream &os, const std::string &metricName,
          const FlexibleMetric &metricValue) {
        os << "<" << metricName << ", "
           << formatFlexibleMetricValue(metricValue.getValues()[0]) << ">";
      };
  std::ostringstream os;
  if (flexibleMetrics.size() == 1) {
    os << scopeName << ": ";
    const auto &[metricName, metricValue] = *flexibleMetrics.begin();
    appendTuple(os, metricName, metricValue);
    return os.str();
  }
  os << scopeName << ": ";
  bool isFirst = true;
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    if (!isFirst) {
      os << ", ";
    }
    appendTuple(os, metricName, metricValue);
    isFirst = false;
  }
  return os.str();
}

std::vector<KernelTrace>
convertToTimelineTrace(std::vector<CycleMetricWithContext> &cycleEvents) {
  std::vector<KernelTrace> results;

  auto getInt64Value = [](const CycleMetric *metric,
                          CycleMetric::CycleMetricKind kind) {
    return std::get<uint64_t>(metric->getValue(kind));
  };

  auto getStringValue = [](const CycleMetric *metric,
                           CycleMetric::CycleMetricKind kind) {
    return std::get<std::string>(metric->getValue(kind));
  };

  auto getKernelId = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::KernelId);
  };

  auto getBlockId = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::BlockId);
  };

  auto getUnitId = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::UnitId);
  };

  auto getStartCycle = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::StartCycle);
  };

  auto getEndCycle = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::EndCycle);
  };

  // Pre-sort all events once
  auto &sortedEvents = cycleEvents;
  std::sort(
      sortedEvents.begin(), sortedEvents.end(),
      [&](const CycleMetricWithContext &a, const CycleMetricWithContext &b) {
        auto aKernelId = getKernelId(a);
        auto bKernelId = getKernelId(b);
        if (aKernelId != bKernelId)
          return aKernelId < bKernelId;

        auto aBlockId = getBlockId(a);
        auto bBlockId = getBlockId(b);
        if (aBlockId != bBlockId)
          return aBlockId < bBlockId;

        auto aUnitId = getUnitId(a);
        auto bUnitId = getUnitId(b);
        if (aUnitId != bUnitId)
          return aUnitId < bUnitId;

        auto aStartCycle = getStartCycle(a);
        auto bStartCycle = getStartCycle(b);
        return aStartCycle < bStartCycle;
      });

  size_t eventIndex = 0;

  // Process in perfectly sorted order
  while (eventIndex < sortedEvents.size()) {
    auto kernelEvent = sortedEvents[eventIndex];
    auto currentKernelId = getKernelId(kernelEvent);

    auto parserResult = std::make_shared<CircularLayoutParserResult>();
    auto metadata = std::make_shared<KernelMetadata>();
    std::map<int, std::string> scopeIdToName;
    std::map<std::string, int> scopeNameToId;
    int curScopeId = 0;
    int64_t timeShiftCost =
        getInt64Value(kernelEvent.cycleMetric, CycleMetric::TimeShiftCost);

    // Process all events for current kernel
    while (eventIndex < sortedEvents.size() &&
           getKernelId(sortedEvents[eventIndex]) == currentKernelId) {

      const auto &blockEvent = sortedEvents[eventIndex];
      uint32_t currentBlockId = getBlockId(blockEvent);
      uint32_t currentProcId =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::ProcessorId);

      CircularLayoutParserResult::BlockTrace blockTrace;
      blockTrace.blockId = currentBlockId;
      blockTrace.procId = currentProcId;
      blockTrace.initTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::InitTime);
      blockTrace.preFinalTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::PreFinalTime);
      blockTrace.postFinalTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::PostFinalTime);
      // Conservative estimation of the number of warps in a CTA.
      blockTrace.traces.reserve(16);

      // Process all events for current block-proc
      while (eventIndex < sortedEvents.size()) {
        const auto &currentEvent = sortedEvents[eventIndex];
        if (getKernelId(currentEvent) != currentKernelId ||
            getBlockId(currentEvent) != currentBlockId) {
          break;
        }

        const auto &uintEvent = sortedEvents[eventIndex];
        uint32_t currentUid = getUnitId(uintEvent);

        CircularLayoutParserResult::Trace unitTrace;
        unitTrace.uid = currentUid;
        // Estimation the number of events in a unit (warp).
        unitTrace.profileEvents.reserve(256);

        // Process all events for current uid
        while (eventIndex < sortedEvents.size()) {
          const auto &event = sortedEvents[eventIndex];
          if (getKernelId(event) != currentKernelId ||
              getBlockId(event) != currentBlockId ||
              getUnitId(event) != currentUid) {
            break;
          }

          auto scopeName = event.contexts.back().name;
          if (scopeNameToId.count(scopeName) == 0) {
            scopeIdToName[curScopeId] = scopeName;
            scopeNameToId[scopeName] = curScopeId;
            curScopeId++;
          }

          auto startEntry = std::make_shared<CycleEntry>();
          startEntry->cycle = getStartCycle(event);
          startEntry->isStart = true;
          startEntry->scopeId = scopeNameToId[scopeName];

          auto endEntry = std::make_shared<CycleEntry>();
          endEntry->cycle = getEndCycle(event);
          endEntry->isStart = false;
          endEntry->scopeId = scopeNameToId[scopeName];

          unitTrace.profileEvents.emplace_back(startEntry, endEntry);

          eventIndex++;
        }
        blockTrace.traces.push_back(std::move(unitTrace));
      }
      parserResult->blockTraces.push_back(std::move(blockTrace));
    }
    std::vector<std::string> callStack;
    if (!sortedEvents.empty()) {
      auto &contexts = kernelEvent.contexts;
      if (!contexts.empty()) {
        callStack.resize(contexts.size() - 1);
        std::transform(contexts.begin(), contexts.end() - 1, callStack.begin(),
                       [](const Context &c) { return c.name; });
      }
    }
    metadata->kernelName =
        getStringValue(kernelEvent.cycleMetric, CycleMetric::KernelName);
    metadata->scopeName = scopeIdToName;
    metadata->callStack = std::move(callStack);
    if (timeShiftCost > 0)
      timeShift(timeShiftCost, parserResult);
    results.emplace_back(parserResult, metadata);
  }
  return results;
}

void dumpCycleMetricTrace(std::vector<CycleMetricWithContext> &cycleEvents,
                          std::ostream &os) {
  auto timeline = convertToTimelineTrace(cycleEvents);
  auto writer = StreamChromeTraceWriter(timeline, "");
  writer.write(os);
}

bool isContextPrefix(const std::vector<Context> &prefix,
                     const std::vector<Context> &full) {
  if (prefix.size() > full.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i].name != full[i].name) {
      return false;
    }
  }
  return true;
}

bool isHiddenGraphTraceContext(const Context &context) {
  return context.name == GraphState::captureTag ||
         context.name == kComputeMetadataScopeName;
}

std::vector<Context>
normalizeGraphKernelContextsForTrace(const std::vector<Context> &contexts) {
  std::vector<Context> normalizedContexts;
  normalizedContexts.reserve(contexts.size());
  for (const auto &context : contexts) {
    if (!isHiddenGraphTraceContext(context)) {
      normalizedContexts.push_back(context);
    }
  }
  return normalizedContexts;
}

std::vector<Context> getGraphScopeContexts(const OrderedTraceEvent &event) {
  if (!event.isGraphLinked || event.contexts.size() <= event.graphScopeStartIndex) {
    return {};
  }
  if (event.contexts.size() <= event.graphScopeStartIndex + 1) {
    return {};
  }
  return std::vector<Context>(event.contexts.begin(), event.contexts.end() - 1);
}

std::vector<std::vector<Context>>
buildGraphScopePrefixes(const std::vector<Context> &scopeContexts,
                        size_t scopeStartIndex) {
  std::vector<std::vector<Context>> prefixes;
  if (scopeStartIndex >= scopeContexts.size()) {
    return prefixes;
  }
  prefixes.reserve(scopeContexts.size() - scopeStartIndex);
  for (size_t i = scopeStartIndex; i < scopeContexts.size(); ++i) {
    prefixes.emplace_back(scopeContexts.begin(), scopeContexts.begin() + i + 1);
  }
  return prefixes;
}

void upsertFlexibleMetric(DataEntry::FlexibleMetricMap &dst,
                          const std::string &metricName,
                          const FlexibleMetric &metric) {
  auto it = dst.find(metricName);
  if (it == dst.end()) {
    dst.emplace(metricName, metric);
  } else {
    it->second.updateValue(metric.getValues()[0]);
  }
}

void mergeFlexibleMetrics(DataEntry::FlexibleMetricMap &dst,
                          const DataEntry::FlexibleMetricMap &src) {
  for (const auto &[metricName, metric] : src) {
    upsertFlexibleMetric(dst, metricName, metric);
  }
}

std::optional<size_t> findGraphScopeRangeIndex(
    const std::map<std::vector<Context>, std::vector<size_t>> &contextToIndices,
    const std::vector<Context> &contexts, uint64_t startTimeNs,
    uint64_t endTimeNs, const std::vector<GraphScopeRange> &graphScopeRanges) {
  auto it = contextToIndices.find(contexts);
  if (it == contextToIndices.end()) {
    return std::nullopt;
  }
  for (auto rangeIndex : it->second) {
    const auto &range = graphScopeRanges[rangeIndex];
    if (range.startTimeNs <= startTimeNs && range.endTimeNs >= endTimeNs) {
      return rangeIndex;
    }
  }
  return std::nullopt;
}

uint64_t getGraphFlowBoundaryStartTimeNs(
    const GraphScopeRange &targetRange, uint64_t kernelStartTimeNs,
    const std::vector<GraphScopeRange> &graphScopeRanges) {
  uint64_t boundaryStartTimeNs = targetRange.startTimeNs;
  for (const auto &candidateRange : graphScopeRanges) {
    if (candidateRange.contexts.size() <= targetRange.contexts.size()) {
      continue;
    }
    if (!isContextPrefix(targetRange.contexts, candidateRange.contexts)) {
      continue;
    }
    if (candidateRange.endTimeNs > kernelStartTimeNs) {
      continue;
    }
    boundaryStartTimeNs = std::max(boundaryStartTimeNs, candidateRange.endTimeNs);
  }
  return boundaryStartTimeNs;
}

std::vector<Context>
normalizeKernelContextsForLaunchScope(const std::vector<Context> &contexts,
                                      const std::vector<Context> &scopeContexts) {
  size_t commonPrefixSize = 0;
  const auto maxCommonPrefixSize =
      std::min(contexts.size(), scopeContexts.size());
  while (commonPrefixSize < maxCommonPrefixSize &&
         contexts[commonPrefixSize].name == scopeContexts[commonPrefixSize].name) {
    ++commonPrefixSize;
  }

  std::vector<Context> normalizedContexts = scopeContexts;
  normalizedContexts.insert(normalizedContexts.end(),
                            contexts.begin() + commonPrefixSize,
                            contexts.end());
  return normalizedContexts;
}

size_t resolveLaunchScopeEventId(
    size_t owningEventId, const std::vector<Context> &kernelContexts,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events,
    const std::unordered_map<size_t, std::vector<Context>> &eventIdToContexts) {
  auto fallbackEventId = TraceData::Trace::TraceEvent::DummyId;
  auto currentEventId = owningEventId;
  while (currentEventId != TraceData::Trace::TraceEvent::DummyId) {
    auto eventIt = events.find(currentEventId);
    if (eventIt == events.end()) {
      break;
    }
    const auto &event = eventIt->second;
    if (event.scopeId != Scope::DummyScopeId && event.hasCpuTimeRange()) {
      fallbackEventId = currentEventId;
      break;
    }
    if (event.launchScopeEventId != TraceData::Trace::TraceEvent::DummyId &&
        event.launchScopeEventId != currentEventId) {
      currentEventId = event.launchScopeEventId;
    } else {
      currentEventId = event.parentEventId;
    }
  }

  const auto owningEventIt = events.find(owningEventId);
  const auto hasOwningCpuStart =
      owningEventIt != events.end() && owningEventIt->second.hasCpuStartTime;
  const auto owningCpuStartTimeNs =
      hasOwningCpuStart ? owningEventIt->second.cpuStartTimeNs : 0;
  const auto owningThreadId =
      owningEventIt != events.end() ? owningEventIt->second.threadId : 0;

  std::vector<Context> candidateScopeContexts = kernelContexts;
  if (!candidateScopeContexts.empty()) {
    candidateScopeContexts.pop_back();
  }
  size_t bestMatchEventId = TraceData::Trace::TraceEvent::DummyId;
  size_t bestMatchDepth = 0;
  for (const auto &[eventId, event] : events) {
    if (event.scopeId == Scope::DummyScopeId || !event.hasCpuTimeRange()) {
      continue;
    }
    if (owningEventIt != events.end() && event.threadId != owningThreadId) {
      continue;
    }
    if (hasOwningCpuStart &&
        (owningCpuStartTimeNs < event.cpuStartTimeNs ||
         owningCpuStartTimeNs > event.cpuEndTimeNs)) {
      continue;
    }
    auto contextsIt = eventIdToContexts.find(eventId);
    if (contextsIt == eventIdToContexts.end() ||
        !isContextPrefix(contextsIt->second, candidateScopeContexts)) {
      continue;
    }
    if (contextsIt->second.size() > bestMatchDepth) {
      bestMatchEventId = eventId;
      bestMatchDepth = contextsIt->second.size();
    }
  }
  if (bestMatchEventId != TraceData::Trace::TraceEvent::DummyId) {
    return bestMatchEventId;
  }
  return fallbackEventId;
}

std::optional<int64_t> computeKernelClockOffsetNs(
    const std::vector<OrderedTraceEvent> &orderedTraceEvents,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events) {
  std::vector<int64_t> offsets;
  offsets.reserve(orderedTraceEvents.size());
  for (const auto &event : orderedTraceEvents) {
    if (event.kind != OrderedTraceEvent::Kind::Kernel || event.isGraphLinked) {
      continue;
    }
    auto traceEventIt = events.find(event.owningEventId);
    if (traceEventIt == events.end() || !traceEventIt->second.hasCpuStartTime) {
      continue;
    }
    offsets.push_back(static_cast<int64_t>(traceEventIt->second.cpuStartTimeNs) -
                      static_cast<int64_t>(event.startTimeNs));
  }
  if (offsets.empty()) {
    return std::nullopt;
  }
  auto middle = offsets.begin() + offsets.size() / 2;
  std::nth_element(offsets.begin(), middle, offsets.end());
  return *middle;
}

uint64_t getAlignedTimestampNs(uint64_t timeNs,
                               const std::optional<int64_t> &clockOffsetNs) {
  if (!clockOffsetNs) {
    return timeNs;
  }
  const auto alignedTimeNs = static_cast<int64_t>(timeNs) + *clockOffsetNs;
  return alignedTimeNs < 0 ? 0 : static_cast<uint64_t>(alignedTimeNs);
}

std::optional<int64_t>
getClockOffsetNs(const OrderedTraceEvent &event,
                 const std::optional<int64_t> &kernelClockOffsetNs) {
  if (event.explicitClockOffsetNs) {
    return event.explicitClockOffsetNs;
  }
  if (event.kind == OrderedTraceEvent::Kind::Kernel) {
    return kernelClockOffsetNs;
  }
  return std::nullopt;
}

uint64_t getAlignedStartTimeNs(const OrderedTraceEvent &event,
                               const std::optional<int64_t> &kernelClockOffsetNs) {
  return getAlignedTimestampNs(event.startTimeNs,
                               getClockOffsetNs(event, kernelClockOffsetNs));
}

std::string getCpuThreadTid(uint64_t threadId) {
  return std::string(kCpuThreadTidPrefix) + std::to_string(threadId);
}

std::string getStreamTid(size_t streamId) {
  return "Stream: " + std::to_string(streamId);
}

std::string getGraphTid(size_t streamId) {
  return "Graph: Stream " + std::to_string(streamId);
}

std::optional<std::pair<size_t, uint64_t>> getCpuFlowSourceScopeEventIdAndTimeNs(
    const OrderedTraceEvent &event,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events) {
  if (event.launchScopeEventId == TraceData::Trace::TraceEvent::DummyId) {
    return std::nullopt;
  }
  auto scopeEventIt = events.find(event.launchScopeEventId);
  if (scopeEventIt == events.end()) {
    return std::nullopt;
  }
  const auto &scopeEvent = scopeEventIt->second;
  if (!scopeEvent.hasCpuTimeRange()) {
    return std::nullopt;
  }
  auto sourceTimeNs = scopeEvent.cpuStartTimeNs;
  if (auto owningEventIt = events.find(event.owningEventId);
      owningEventIt != events.end() && owningEventIt->second.hasCpuStartTime) {
    sourceTimeNs = owningEventIt->second.cpuStartTimeNs;
  }
  if (scopeEvent.hasCpuTimeRange()) {
    sourceTimeNs =
        std::clamp(sourceTimeNs, scopeEvent.cpuStartTimeNs, scopeEvent.cpuEndTimeNs);
  }
  return std::make_pair(event.launchScopeEventId, sourceTimeNs);
}

uint64_t getAdjustedFlowStartTimeNs(
    uint64_t sourceTimeNs, uint64_t alignedKernelStartTimeNs,
    const TraceData::Trace::TraceEvent &scopeEvent) {
  auto latestFlowStartTimeNs =
      std::min(scopeEvent.cpuEndTimeNs, alignedKernelStartTimeNs);
  if (latestFlowStartTimeNs > scopeEvent.cpuStartTimeNs) {
    latestFlowStartTimeNs -= 1;
  }
  if (latestFlowStartTimeNs < scopeEvent.cpuStartTimeNs) {
    return scopeEvent.cpuStartTimeNs;
  }
  return std::clamp(sourceTimeNs, scopeEvent.cpuStartTimeNs,
                    latestFlowStartTimeNs);
}

uint64_t getAdjustedGraphFlowStartTimeNs(uint64_t alignedGraphScopeStartTimeNs,
                                         uint64_t alignedGraphScopeEndTimeNs,
                                         uint64_t alignedKernelStartTimeNs) {
  auto latestFlowStartTimeNs =
      std::min(alignedGraphScopeEndTimeNs, alignedKernelStartTimeNs);
  if (latestFlowStartTimeNs < alignedGraphScopeStartTimeNs) {
    return alignedGraphScopeStartTimeNs;
  }
  auto earliestFlowStartTimeNs = alignedGraphScopeStartTimeNs;
  if (earliestFlowStartTimeNs < latestFlowStartTimeNs) {
    earliestFlowStartTimeNs += 1;
  }
  return std::min(earliestFlowStartTimeNs, latestFlowStartTimeNs);
}

void reconstructGraphScopeEvents(
    std::vector<OrderedTraceEvent> &orderedTraceEvents,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events) {
  using GroupKey = std::pair<size_t, size_t>;
  std::map<GroupKey, std::vector<GraphKernelRecord>> groupToGraphKernels;
  std::map<size_t, GraphKernelRecord> owningEventIdToFirstGraphKernel;
  for (size_t i = 0; i < orderedTraceEvents.size(); ++i) {
    const auto &event = orderedTraceEvents[i];
    if (event.kind != OrderedTraceEvent::Kind::Kernel || !event.isGraphLinked) {
      continue;
    }
    auto scopeContexts = getGraphScopeContexts(event);
    if (scopeContexts.empty()) {
      continue;
    }
    auto record =
        GraphKernelRecord{i, event.owningEventId, event.streamId, event.startTimeNs,
                          event.endTimeNs, event.graphScopeStartIndex,
                          std::move(scopeContexts),
                          event.getFlexibleMetrics()};
    groupToGraphKernels[{event.owningEventId, event.streamId}].push_back(record);
    auto firstGraphKernelIt = owningEventIdToFirstGraphKernel.find(event.owningEventId);
    if (firstGraphKernelIt == owningEventIdToFirstGraphKernel.end() ||
        record.startTimeNs < firstGraphKernelIt->second.startTimeNs ||
        (record.startTimeNs == firstGraphKernelIt->second.startTimeNs &&
         record.endTimeNs < firstGraphKernelIt->second.endTimeNs) ||
        (record.startTimeNs == firstGraphKernelIt->second.startTimeNs &&
         record.endTimeNs == firstGraphKernelIt->second.endTimeNs &&
         record.orderedEventIndex < firstGraphKernelIt->second.orderedEventIndex)) {
      owningEventIdToFirstGraphKernel[event.owningEventId] = record;
    }
  }

  std::map<size_t, std::optional<int64_t>> owningEventIdToClockOffsetNs;
  for (const auto &[owningEventId, firstGraphKernel] : owningEventIdToFirstGraphKernel) {
    const auto &firstKernelEvent =
        orderedTraceEvents[firstGraphKernel.orderedEventIndex];
    std::optional<uint64_t> anchorTimeNs;
    if (auto flowSource =
            getCpuFlowSourceScopeEventIdAndTimeNs(firstKernelEvent, events);
        flowSource) {
      anchorTimeNs = flowSource->second;
    } else if (auto traceEventIt = events.find(owningEventId);
               traceEventIt != events.end() &&
               traceEventIt->second.hasCpuStartTime) {
      anchorTimeNs = traceEventIt->second.cpuStartTimeNs;
    }
    if (anchorTimeNs) {
      owningEventIdToClockOffsetNs[owningEventId] =
          static_cast<int64_t>(*anchorTimeNs) -
          static_cast<int64_t>(firstGraphKernel.startTimeNs);
    }
  }

  std::vector<OrderedTraceEvent> graphScopeEvents;
  for (auto &[_, graphKernelRecords] : groupToGraphKernels) {
    std::sort(graphKernelRecords.begin(), graphKernelRecords.end(),
              [](const GraphKernelRecord &lhs, const GraphKernelRecord &rhs) {
                if (lhs.startTimeNs != rhs.startTimeNs) {
                  return lhs.startTimeNs < rhs.startTimeNs;
                }
                if (lhs.endTimeNs != rhs.endTimeNs) {
                  return lhs.endTimeNs < rhs.endTimeNs;
                }
                return lhs.orderedEventIndex < rhs.orderedEventIndex;
              });

    const auto graphClockOffsetIt =
        owningEventIdToClockOffsetNs.find(graphKernelRecords.front().owningEventId);
    const auto graphClockOffsetNs =
        graphClockOffsetIt != owningEventIdToClockOffsetNs.end()
            ? graphClockOffsetIt->second
            : std::nullopt;

    std::vector<OpenGraphScope> openScopes;
    std::vector<GraphScopeRange> graphScopeRanges;
    uint64_t previousEndTimeNs = 0;
    bool hasPreviousRange = false;

    for (const auto &record : graphKernelRecords) {
      auto scopePrefixes =
          buildGraphScopePrefixes(record.scopeContexts, record.scopeStartIndex);
      if (scopePrefixes.empty()) {
        continue;
      }

      size_t commonPrefixSize = 0;
      while (commonPrefixSize < openScopes.size() &&
             commonPrefixSize < scopePrefixes.size() &&
             openScopes[commonPrefixSize].contexts ==
                 scopePrefixes[commonPrefixSize]) {
        ++commonPrefixSize;
      }

      if (hasPreviousRange) {
        while (openScopes.size() > commonPrefixSize) {
          auto openScope = std::move(openScopes.back());
          openScopes.pop_back();
          graphScopeRanges.push_back(GraphScopeRange{
              std::move(openScope.contexts), record.streamId,
              openScope.startTimeNs, previousEndTimeNs, {}});
        }
      }

      for (size_t i = commonPrefixSize; i < scopePrefixes.size(); ++i) {
        openScopes.push_back(
            OpenGraphScope{std::move(scopePrefixes[i]), record.startTimeNs});
      }
      previousEndTimeNs = record.endTimeNs;
      hasPreviousRange = true;
    }

    while (!openScopes.empty()) {
      auto openScope = std::move(openScopes.back());
      openScopes.pop_back();
      graphScopeRanges.push_back(GraphScopeRange{
          std::move(openScope.contexts), graphKernelRecords.front().streamId,
          openScope.startTimeNs, previousEndTimeNs, {}});
    }

    std::sort(graphScopeRanges.begin(), graphScopeRanges.end(),
              [](const GraphScopeRange &lhs, const GraphScopeRange &rhs) {
                if (lhs.startTimeNs != rhs.startTimeNs) {
                  return lhs.startTimeNs < rhs.startTimeNs;
                }
                if (lhs.endTimeNs != rhs.endTimeNs) {
                  return lhs.endTimeNs > rhs.endTimeNs;
                }
                return lhs.contexts.size() < rhs.contexts.size();
              });

    std::map<std::vector<Context>, std::vector<size_t>> contextToIndices;
    for (size_t i = 0; i < graphScopeRanges.size(); ++i) {
      contextToIndices[graphScopeRanges[i].contexts].push_back(i);
    }

    for (const auto &record : graphKernelRecords) {
      auto rangeIndex = findGraphScopeRangeIndex(
          contextToIndices, record.scopeContexts, record.startTimeNs,
          record.endTimeNs, graphScopeRanges);
      if (!rangeIndex) {
        continue;
      }
      auto &kernelEvent = orderedTraceEvents[record.orderedEventIndex];
      auto &graphScopeRange = graphScopeRanges[*rangeIndex];
      const auto graphFlowBoundaryStartTimeNs = getGraphFlowBoundaryStartTimeNs(
          graphScopeRange, record.startTimeNs, graphScopeRanges);
      kernelEvent.explicitClockOffsetNs = graphClockOffsetNs;
      kernelEvent.hasGraphFlowSource = true;
      kernelEvent.graphFlowStartTimeNs = graphFlowBoundaryStartTimeNs;
      kernelEvent.graphFlowEndTimeNs = graphScopeRange.endTimeNs;
      if (kernelEvent.isMetricKernel && record.flexibleMetrics &&
          !record.flexibleMetrics->empty()) {
        mergeFlexibleMetrics(graphScopeRange.flexibleMetrics,
                             *record.flexibleMetrics);
      }
    }

    for (auto &graphScopeRange : graphScopeRanges) {
      std::shared_ptr<DataEntry::FlexibleMetricMap> flexibleMetricsStorage;
      if (!graphScopeRange.flexibleMetrics.empty()) {
        flexibleMetricsStorage = std::make_shared<DataEntry::FlexibleMetricMap>(
            std::move(graphScopeRange.flexibleMetrics));
      }
      graphScopeEvents.push_back(OrderedTraceEvent::graphScope(
          std::move(graphScopeRange.contexts), graphScopeRange.streamId,
          graphScopeRange.startTimeNs, graphScopeRange.endTimeNs,
          std::move(flexibleMetricsStorage), graphClockOffsetNs));
    }
  }

  orderedTraceEvents.insert(orderedTraceEvents.end(),
                            std::make_move_iterator(graphScopeEvents.begin()),
                            std::make_move_iterator(graphScopeEvents.end()));
}

void dumpKernelMetricTrace(
    const std::vector<OrderedTraceEvent> &orderedTraceEvents,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events,
    std::ostream &os) {
  json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
  const bool hasCpuScopeEvents = std::any_of(
      orderedTraceEvents.begin(), orderedTraceEvents.end(),
      [](const auto &event) {
        return event.kind == OrderedTraceEvent::Kind::CpuScope;
      });
  const auto kernelClockOffsetNs =
      hasCpuScopeEvents ? computeKernelClockOffsetNs(orderedTraceEvents, events)
                         : std::nullopt;
  auto minTimeStamp = std::numeric_limits<uint64_t>::max();
  for (const auto &event : orderedTraceEvents) {
    minTimeStamp =
        std::min(minTimeStamp, getAlignedStartTimeNs(event, kernelClockOffsetNs));
    if (event.kind == OrderedTraceEvent::Kind::Kernel) {
      const auto alignedStartTimeNs =
          getAlignedStartTimeNs(event, kernelClockOffsetNs);
      if (event.hasGraphFlowSource) {
        const auto alignedGraphScopeStartTimeNs = getAlignedTimestampNs(
            event.graphFlowStartTimeNs,
            getClockOffsetNs(event, kernelClockOffsetNs));
        const auto alignedGraphScopeEndTimeNs = getAlignedTimestampNs(
            event.graphFlowEndTimeNs, getClockOffsetNs(event, kernelClockOffsetNs));
        const auto adjustedGraphFlowStartTimeNs = getAdjustedGraphFlowStartTimeNs(
            alignedGraphScopeStartTimeNs, alignedGraphScopeEndTimeNs,
            alignedStartTimeNs);
        minTimeStamp =
            std::min(minTimeStamp,
                     std::min(adjustedGraphFlowStartTimeNs, alignedStartTimeNs));
      } else if (auto flowSource =
                     getCpuFlowSourceScopeEventIdAndTimeNs(event, events);
                 flowSource) {
        const auto &scopeEvent = events.at(flowSource->first);
        minTimeStamp = std::min(
            minTimeStamp, getAdjustedFlowStartTimeNs(flowSource->second,
                                                     alignedStartTimeNs,
                                                     scopeEvent));
      }
    }
  }
  uint64_t nextFlowId = 0;
  auto buildScopeTraceElement = [&](const OrderedTraceEvent &event) {
    json element;
    const bool eventHasFlexibleMetrics = hasFlexibleMetrics(event);
    element["name"] = eventHasFlexibleMetrics
                          ? buildFlexibleMetricEventName(
                                event.contexts, *event.getFlexibleMetrics())
                          : event.contexts.back().name;
    element["cat"] = eventHasFlexibleMetrics ? "metric" : "scope";
    if (event.kind == OrderedTraceEvent::Kind::CpuScope) {
      element["tid"] = getCpuThreadTid(event.threadId);
      element["args"]["thread_id"] = event.threadId;
    } else {
      element["tid"] = getGraphTid(event.streamId);
    }
    element["args"]["call_stack"] = buildCallStackJson(event.contexts);
    if (eventHasFlexibleMetrics) {
      element["args"]["metrics"] =
          buildFlexibleMetricsJson(*event.getFlexibleMetrics());
    }
    return element;
  };
  auto appendTraceEvent = [&](const OrderedTraceEvent &event) {
    if (event.kind == OrderedTraceEvent::Kind::GraphScope) {
      return;
    }
    if (event.kind == OrderedTraceEvent::Kind::CpuScope && event.contexts.empty()) {
      return;
    }

    json element;
    if (event.kind == OrderedTraceEvent::Kind::Kernel) {
      if (event.isMetricKernel) {
        element["name"] = GraphState::metricTag;
      } else {
        element["name"] = event.contexts.back().name;
      }
      element["cat"] = "kernel";
      element["tid"] = getStreamTid(event.streamId);
    } else {
      element = buildScopeTraceElement(event);
    }
    const auto alignedStartTimeNs =
        getAlignedStartTimeNs(event, kernelClockOffsetNs);
    element["ph"] = "X";
    element["ts"] =
        static_cast<double>(alignedStartTimeNs - minTimeStamp) / 1000.0;
    element["dur"] =
        static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
    object["traceEvents"].push_back(element);
    if (event.kind != OrderedTraceEvent::Kind::Kernel) {
      return;
    }
    if (event.hasGraphFlowSource) {
      const auto alignedGraphScopeStartTimeNs = getAlignedTimestampNs(
          event.graphFlowStartTimeNs, getClockOffsetNs(event, kernelClockOffsetNs));
      const auto alignedGraphScopeEndTimeNs = getAlignedTimestampNs(
          event.graphFlowEndTimeNs, getClockOffsetNs(event, kernelClockOffsetNs));
      const auto adjustedGraphFlowStartTimeNs = getAdjustedGraphFlowStartTimeNs(
          alignedGraphScopeStartTimeNs, alignedGraphScopeEndTimeNs,
          alignedStartTimeNs);
      json flowStart = {{"name", kLaunchFlowName},
                        {"cat", kLaunchFlowCategory},
                        {"ph", "s"},
                        {"bp", "e"},
                        {"id", nextFlowId},
                        {"ts",
                         static_cast<double>(adjustedGraphFlowStartTimeNs -
                                             minTimeStamp) /
                             1000.0},
                        {"tid", getGraphTid(event.streamId)}};
      json flowFinish = {{"name", kLaunchFlowName},
                         {"cat", kLaunchFlowCategory},
                         {"ph", "f"},
                         {"bp", "e"},
                         {"id", nextFlowId},
                         {"ts",
                          static_cast<double>(alignedStartTimeNs - minTimeStamp) /
                              1000.0},
                         {"tid", getStreamTid(event.streamId)}};
      object["traceEvents"].push_back(flowStart);
      object["traceEvents"].push_back(flowFinish);
      ++nextFlowId;
      return;
    }
    const auto flowSource = getCpuFlowSourceScopeEventIdAndTimeNs(event, events);
    if (!flowSource) {
      return;
    }
    const auto [metricScopeEventId, sourceTimeNs] = *flowSource;
    const auto &scopeEvent = events.at(metricScopeEventId);
    const auto adjustedFlowStartTimeNs =
        getAdjustedFlowStartTimeNs(sourceTimeNs, alignedStartTimeNs, scopeEvent);
    json flowStart = {{"name", kLaunchFlowName},
                      {"cat", kLaunchFlowCategory},
                      {"ph", "s"},
                      {"bp", "e"},
                      {"id", nextFlowId},
                      {"ts",
                       static_cast<double>(adjustedFlowStartTimeNs - minTimeStamp) /
                           1000.0},
                      {"tid", getCpuThreadTid(scopeEvent.threadId)}};
    json flowFinish = {{"name", kLaunchFlowName},
                       {"cat", kLaunchFlowCategory},
                       {"ph", "f"},
                       {"bp", "e"},
                       {"id", nextFlowId},
                       {"ts",
                        static_cast<double>(alignedStartTimeNs - minTimeStamp) /
                            1000.0},
                       {"tid", getStreamTid(event.streamId)}};
    object["traceEvents"].push_back(flowStart);
    object["traceEvents"].push_back(flowFinish);
    ++nextFlowId;
  };
  std::vector<GraphScopeBoundary> graphScopeBoundaries;
  graphScopeBoundaries.reserve(orderedTraceEvents.size() * 2);
  size_t nextGraphScopeBoundaryOrder = 0;
  for (const auto &event : orderedTraceEvents) {
    if (event.kind != OrderedTraceEvent::Kind::GraphScope || event.contexts.empty()) {
      continue;
    }
    const auto clockOffsetNs = getClockOffsetNs(event, kernelClockOffsetNs);
    graphScopeBoundaries.push_back(
        GraphScopeBoundary{&event,
                           getAlignedTimestampNs(event.startTimeNs, clockOffsetNs),
                           true,
                           nextGraphScopeBoundaryOrder++});
    graphScopeBoundaries.push_back(
        GraphScopeBoundary{&event,
                           getAlignedTimestampNs(event.endTimeNs, clockOffsetNs),
                           false,
                           nextGraphScopeBoundaryOrder++});
  }
  std::sort(graphScopeBoundaries.begin(), graphScopeBoundaries.end(),
            [](const GraphScopeBoundary &lhs, const GraphScopeBoundary &rhs) {
              if (lhs.alignedTimeNs != rhs.alignedTimeNs) {
                return lhs.alignedTimeNs < rhs.alignedTimeNs;
              }
              if (lhs.isBegin != rhs.isBegin) {
                return !lhs.isBegin && rhs.isBegin;
              }
              if (lhs.event->contexts.size() != rhs.event->contexts.size()) {
                if (lhs.isBegin) {
                  return lhs.event->contexts.size() < rhs.event->contexts.size();
                }
                return lhs.event->contexts.size() > rhs.event->contexts.size();
              }
              if (lhs.event->contexts != rhs.event->contexts) {
                return lhs.event->contexts < rhs.event->contexts;
              }
              return lhs.insertionOrder < rhs.insertionOrder;
            });
  auto appendGraphScopeBoundary = [&](const GraphScopeBoundary &boundary) {
    json element = buildScopeTraceElement(*boundary.event);
    element["ph"] = boundary.isBegin ? "B" : "E";
    element["ts"] =
        static_cast<double>(boundary.alignedTimeNs - minTimeStamp) / 1000.0;
    object["traceEvents"].push_back(element);
  };

  for (const auto &event : orderedTraceEvents) {
    if (event.kind == OrderedTraceEvent::Kind::CpuScope) {
      appendTraceEvent(event);
    }
  }
  for (const auto &boundary : graphScopeBoundaries) {
    appendGraphScopeBoundary(boundary);
  }
  for (const auto &event : orderedTraceEvents) {
    if (event.kind == OrderedTraceEvent::Kind::Kernel) {
      appendTraceEvent(event);
    }
  }

  os << object.dump() << "\n";
}

} // namespace

void TraceData::dumpChromeTrace(std::ostream &os, size_t phase) const {
  std::set<size_t> virtualTargetEntryIds;
  tracePhases.withPtr(phase, [&](Trace *trace) {
    for (const auto &[_, event] : trace->getEvents()) {
      for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
        virtualTargetEntryIds.insert(targetEntryId);
      }
      for (const auto &[targetEntryId, _] :
           event.metricSet.linkedFlexibleMetrics) {
        virtualTargetEntryIds.insert(targetEntryId);
      }
    }
  });

  std::map<size_t, std::vector<Context>> targetIdToVirtualContexts;
  if (!virtualTargetEntryIds.empty()) {
    tracePhases.withPtr(Data::kVirtualPhase, [&](Trace *virtualTrace) {
      for (auto targetEntryId : virtualTargetEntryIds) {
        // Linked target ids are event ids, so resolve through the event first.
        auto &targetEvent = virtualTrace->getEvent(targetEntryId);
        auto contexts = virtualTrace->getContexts(targetEvent.contextId);
        contexts.erase(contexts.begin());
        targetIdToVirtualContexts.emplace(targetEntryId, std::move(contexts));
      }
    });
  }

  tracePhases.withPtr(phase, [&](Trace *trace) {
    auto &events = trace->getEvents();
    std::vector<OrderedTraceEvent> orderedTraceEvents;
    orderedTraceEvents.reserve(events.size() * 2);
    std::unordered_map<size_t, std::vector<Context>> eventIdToContexts;
    eventIdToContexts.reserve(events.size());
    for (const auto &[eventId, event] : events) {
      eventIdToContexts.emplace(eventId, trace->getContexts(event.contextId));
    }
    bool hasKernelMetrics = false, hasCycleMetrics = false,
         hasCpuScopeEvents = false;
    std::vector<CycleMetricWithContext> cycleEvents;
    cycleEvents.reserve(events.size());

    auto processMetricMaps =
        [&](size_t owningEventId, const DataEntry::MetricMap &metrics,
            const DataEntry::FlexibleMetricMap *flexibleMetrics,
            const std::vector<Context> &contexts, bool isGraphLinked) {
          bool emittedKernel = false;
          if (auto kernelIt = metrics.find(MetricKind::Kernel);
              kernelIt != metrics.end()) {
            auto *kernelMetric =
                static_cast<KernelMetric *>(kernelIt->second.get());
            const auto isMetricKernel = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::IsMetricKernel));
            const auto streamId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StreamId));
            const auto launchScopeEventId =
                resolveLaunchScopeEventId(owningEventId, contexts, events,
                                          eventIdToContexts);
            auto kernelContexts = contexts;
            size_t graphScopeStartIndex = 0;
            if (launchScopeEventId != TraceData::Trace::TraceEvent::DummyId) {
              auto launchScopeContextsIt =
                  eventIdToContexts.find(launchScopeEventId);
              if (launchScopeContextsIt != eventIdToContexts.end()) {
                graphScopeStartIndex = launchScopeContextsIt->second.size();
                kernelContexts = normalizeKernelContextsForLaunchScope(
                    contexts, launchScopeContextsIt->second);
              }
            }
            if (isGraphLinked) {
              kernelContexts = normalizeGraphKernelContextsForTrace(kernelContexts);
            }
            if (isMetricKernel) {
              orderedTraceEvents.push_back(OrderedTraceEvent::kernel(
                  kernelMetric, flexibleMetrics, kernelContexts, owningEventId,
                  streamId, launchScopeEventId, isGraphLinked,
                  isMetricKernel, graphScopeStartIndex));
            } else {
              orderedTraceEvents.push_back(OrderedTraceEvent::kernel(
                  kernelMetric, nullptr, kernelContexts, owningEventId, streamId,
                  launchScopeEventId, isGraphLinked, isMetricKernel,
                  graphScopeStartIndex));
            }
            hasKernelMetrics = true;
            emittedKernel = true;
          }
          if (auto cycleIt = metrics.find(MetricKind::Cycle);
              cycleIt != metrics.end()) {
            auto *cycleMetric =
                static_cast<CycleMetric *>(cycleIt->second.get());
            cycleEvents.emplace_back(cycleMetric, contexts);
            hasCycleMetrics = true;
          }
          return emittedKernel;
        };

    for (const auto &[_, event] : events) {
      const auto &baseContexts = eventIdToContexts.at(event.id);
      processMetricMaps(event.id, event.metricSet.metrics,
                        &event.metricSet.flexibleMetrics, baseContexts,
                        /*isGraphLinked=*/false);
      std::vector<size_t> sortedLinkedTargetEntryIds;
      sortedLinkedTargetEntryIds.reserve(event.metricSet.linkedMetrics.size());
      for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
        sortedLinkedTargetEntryIds.push_back(targetEntryId);
      }
      std::sort(sortedLinkedTargetEntryIds.begin(),
                sortedLinkedTargetEntryIds.end());
      for (const auto targetEntryId : sortedLinkedTargetEntryIds) {
        const auto &linkedMetrics =
            event.metricSet.linkedMetrics.at(targetEntryId);
        auto contexts = baseContexts;
        auto &virtualContexts = targetIdToVirtualContexts[targetEntryId];
        contexts.insert(contexts.end(), virtualContexts.begin(),
                        virtualContexts.end());
        const DataEntry::FlexibleMetricMap *flexibleMetrics = nullptr;
        auto iter = event.metricSet.linkedFlexibleMetrics.find(targetEntryId);
        if (iter != event.metricSet.linkedFlexibleMetrics.end()) {
          flexibleMetrics = &iter->second;
        }
        processMetricMaps(event.id, linkedMetrics, flexibleMetrics, contexts,
                          /*isGraphLinked=*/true);
      }
      if (hasKernelMetrics && hasCycleMetrics) {
        throw std::runtime_error("only one active metric type is supported");
      }
    }
    for (const auto &[eventId, event] : events) {
      if (event.scopeId == Scope::DummyScopeId || !event.hasCpuTimeRange()) {
        continue;
      }
      orderedTraceEvents.push_back(OrderedTraceEvent::cpuScope(
          event.metricSet.flexibleMetrics.empty() ? nullptr
                                                  : &event.metricSet.flexibleMetrics,
          eventIdToContexts.at(eventId), event.threadId, event.cpuStartTimeNs,
          event.cpuEndTimeNs));
      hasCpuScopeEvents = true;
    }

    reconstructGraphScopeEvents(orderedTraceEvents, events);

    if (hasCycleMetrics) {
      dumpCycleMetricTrace(cycleEvents, os);
      return;
    }

    if (hasKernelMetrics || hasCpuScopeEvents) {
      dumpKernelMetricTrace(orderedTraceEvents, events, os);
    } else {
      os << json({{"displayTimeUnit", "us"}, {"traceEvents", json::array()}})
                .dump()
         << "\n";
    }
  });
}

void TraceData::doDump(std::ostream &os, OutputFormat outputFormat,
                       size_t phase) const {
  if (outputFormat == OutputFormat::ChromeTrace) {
    dumpChromeTrace(os, phase);
  } else {
    throw std::logic_error("Output format not supported");
  }
}

TraceData::TraceData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(tracePhases);
}

TraceData::~TraceData() {}

} // namespace proton
