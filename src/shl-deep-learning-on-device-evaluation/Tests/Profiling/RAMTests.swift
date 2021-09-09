import XCTest

@testable import SHLModelEvaluation

class RAMTests: XCTestCase {
    let modelIds = [
        "6af8a156f95b675b9ae9689d66c25b64",
        "968ca3e771b5efa5855ba841f69c463a",
        "7523152748da6ca5d4d59234b3732c89",
        "4ad2e771492aac5b42c89d9b5a835857",
        "e6c36740f115fc7510ff472225bedeac",
        "35d1710beb7573fa3d55a684e5e5d9a5",
        "7cfd6602ff1ee897fd95deebf54ec24a",
        "ea70373fe386b068ffab6704b4ac25fc",
        "9988226d7297eb039365ad50f7c5ea0b",
        "cb5048dd197b94c8b420dbcdb49f378f",
    ]

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
        for modelId in modelIds {
            let classifier = try Classifier(
                modelFileName: modelId,
                accelerator: .none,
                threads: 1
            )
            // Run
            try run(classifier: classifier, runs: 3)
            let mb = ram_usage()

            print("CPU - RAM Usage for \(modelId): \(mb)")
        }
    }

    func testGPU() throws {
        for modelId in modelIds {
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
        for modelId in modelIds {
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

