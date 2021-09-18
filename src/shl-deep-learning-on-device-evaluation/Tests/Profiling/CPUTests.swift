import XCTest

@testable import SHLModelEvaluation

typealias CPUUsage = Float

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
            usages.append(cpu_usage())
            usleep(500_000)
            usages.append(cpu_usage())
        }
        return usages.reduce(0, +) / Float(usages.count)
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

