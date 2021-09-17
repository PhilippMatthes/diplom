import XCTest

@testable import SHLModelEvaluation

typealias CPUUsage = Double

fileprivate func cpuUsage() -> CPUUsage {
    var kr: kern_return_t
    var task_info_count: mach_msg_type_number_t

    task_info_count = mach_msg_type_number_t(TASK_INFO_MAX)
    var tinfo = [integer_t](repeating: 0, count: Int(task_info_count))

    kr = task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO), &tinfo, &task_info_count)
    if kr != KERN_SUCCESS {
        return -1
    }

    var thread_list: thread_act_array_t? = UnsafeMutablePointer(mutating: [thread_act_t]())
    var thread_count: mach_msg_type_number_t = 0
    defer {
        if let thread_list = thread_list {
            vm_deallocate(mach_task_self_, vm_address_t(UnsafePointer(thread_list).pointee), vm_size_t(thread_count))
        }
    }

    kr = task_threads(mach_task_self_, &thread_list, &thread_count)

    if kr != KERN_SUCCESS {
        return -1
    }

    var tot_cpu: Double = 0

    if let thread_list = thread_list {

        for j in 0 ..< Int(thread_count) {
            var thread_info_count = mach_msg_type_number_t(THREAD_INFO_MAX)
            var thinfo = [integer_t](repeating: 0, count: Int(thread_info_count))
            kr = thread_info(thread_list[j], thread_flavor_t(THREAD_BASIC_INFO),
                             &thinfo, &thread_info_count)
            if kr != KERN_SUCCESS {
                return -1
            }

            let threadBasicInfo = convertThreadInfoToThreadBasicInfo(thinfo)

            if threadBasicInfo.flags != TH_FLAGS_IDLE {
                tot_cpu += (Double(threadBasicInfo.cpu_usage) / Double(TH_USAGE_SCALE)) * 100.0
            }
        } // for each thread
    }

    return tot_cpu
}

fileprivate func convertThreadInfoToThreadBasicInfo(_ threadInfo: [integer_t]) -> thread_basic_info {
    var result = thread_basic_info()

    result.user_time = time_value_t(seconds: threadInfo[0], microseconds: threadInfo[1])
    result.system_time = time_value_t(seconds: threadInfo[2], microseconds: threadInfo[3])
    result.cpu_usage = threadInfo[4]
    result.policy = threadInfo[5]
    result.run_state = threadInfo[6]
    result.flags = threadInfo[7]
    result.suspend_count = threadInfo[8]
    result.sleep_time = threadInfo[9]

    return result
}

class CPUTests: XCTestCase {
    @discardableResult private func run(classifier: Classifier, runs: Int) throws -> CPUUsage {
        var usages = [CPUUsage]()
        for _ in (0..<runs) {
            let randomInputs = Array((0..<500)).map { i -> [Parameter] in
                Array((0..<3)).map { _ -> Parameter in
                    Parameter.random(in: -1.0 ... 1.0)
                }
            }
            _ = try classifier.classify(input: randomInputs)
            usages.append(cpuUsage())
            usleep(500_000)
            usages.append(cpuUsage())
        }
        return usages.reduce(0, +) / Double(usages.count)
    }

    func testCPU() throws {
        for modelId in Models.all {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .cpu,
                threads: 1
            )
            // Cold run
            try run(classifier: classifier, runs: 3)

            // Warm run
            let usage = try run(classifier: classifier, runs: 10)

            print("CPU - CPU Usage for \(modelId): \(usage)")
        }
    }

    func testGPU() throws {
        for modelId in Models.all {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .gpu,
                threads: 1
            )
            // Cold run
            try run(classifier: classifier, runs: 3)

            // Warm run
            let usage = try run(classifier: classifier, runs: 10)

            print("GPU - CPU Usage for \(modelId): \(usage)")
        }
    }

    /// Test the ANE.
    func testANE() throws {
        for modelId in Models.all {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .ane,
                threads: 1
            )
            // Cold run
            try run(classifier: classifier, runs: 3)

            // Warm run
            let usage = try run(classifier: classifier, runs: 10)

            print("ANE - CPU Usage for \(modelId): \(usage)")
        }
    }
}

