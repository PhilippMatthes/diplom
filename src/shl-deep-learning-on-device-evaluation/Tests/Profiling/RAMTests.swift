import XCTest

@testable import SHLModelEvaluation

class RAMTests: XCTest {
    private func run(classifier: Classifier, runs: Int) throws {
        for _ in (0..<runs) {
            let randomInputs = Array((0..<500)).map { i -> [Parameter] in
                Array((0..<3)).map { _ -> Parameter in
                    Parameter.random(in: -1.0 ... 1.0)
                }
            }
            _ = try classifier.classify(input: randomInputs)
        }
    }

    func testCPU() throws {
        for modelId in Models.all {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .cpu,
                threads: 1
            )
            // Run
            try run(classifier: classifier, runs: 3)
            let mb = ram_usage()

            print("CPU - RAM Usage for \(modelId): \(mb)")
        }
    }

    func testGPU() throws {
        for modelId in Models.all {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .gpu,
                threads: 1
            )
            // Run
            try run(classifier: classifier, runs: 3)
            let mb = ram_usage()

            print("GPU - RAM Usage for \(modelId): \(mb)")
        }
    }

    func testANE() throws {
        for modelId in Models.all {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .ane,
                threads: 1
            )
            // Run
            try run(classifier: classifier, runs: 3)
            let mb = ram_usage()

            print("ANE - RAM Usage for \(modelId): \(mb)")
        }
    }
}

