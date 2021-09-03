import Foundation

@testable import SHLModelEvaluation

class ModelTests: XCTestCase {
    func testInferenceTime() throws {
        let classifier = try Classifier(modelFileName: "model")
        measure {
            let numberOfRuns = 100
            for r in (0..<numberOfRuns) {
                let randomInputs = Array((0..<3)).map { i -> [Parameter] in
                    Array((0..<500)).map { _ -> Parameter in
                        Parameter.random(in: -1.0 ... 1.0)
                    }
                }
                let predictions = try! classifier.classify(input: randomInputs)
                print("\(r)/\(numberOfRuns) -> \(predictions[0].label) w. \(predictions[0].confidence) confidence")
            }
        }
    }
}
