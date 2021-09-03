import XCTest

@testable import SHLModelEvaluation

private struct TestSample: Codable {
    let xFeature: [[Parameter]]
    let xRaw: [[Parameter]]
    let y: Classifier.Class

    private enum CodingKeys: String, CodingKey {
        case xFeature = "X_feature"
        case xRaw = "X_raw"
        case y = "y"
    }
}

class PreprocessingTests: XCTestCase {
    private var datasets: [String: [TestSample]]?

    override func setUp() {
        super.setUp()
        guard
            let testSampleUrl = Bundle.main.url(
                forResource: "testdata", withExtension: "json"
            ),
            let data = try? Data(
                contentsOf: testSampleUrl, options: .mappedIfSafe
            ),
            let datasets = try? JSONDecoder().decode(
                [String: [TestSample]].self, from: data
            )
        else { XCTFail(); return }
        XCTAssert(!datasets.isEmpty)
        self.datasets = datasets
    }

    /// Test that the preprocessors were ported correctly.
    func testPreprocessors() throws {
        guard let datasets = datasets else { XCTFail(); return }

        let sensorPreprocessors = Dictionary(uniqueKeysWithValues: try Sensor.order.map { sensor throws -> (Sensor, [Preprocessor]) in
            let preprocessors: [Preprocessor] = [
                try PowerTransformer(configFileName: sensor.configFileName),
                try StandardScaler(configFileName: sensor.configFileName),
            ]
            return (sensor, preprocessors)
        })

        for sensor in Sensor.order {
            print("Testing preprocessors for sensor \(sensor.description)")
            for sample in datasets.values.flatMap({ s in s }) {
                guard
                    let preprocessors = sensorPreprocessors[sensor]
                else { XCTFail(); return }

                let xFeatureSensor = sample.xFeature
                    .map { features in features[sensor.sampleIndex] }
                let xRawSensor = sample.xRaw
                    .map { features in features[sensor.sampleIndex] }
                var xPreprocessed = xRawSensor
                for preprocessor in preprocessors {
                    xPreprocessed = preprocessor.transform(xPreprocessed)
                }

                let xDiffs = Array((0..<500)).map { i in
                    xFeatureSensor[i] - xPreprocessed[i]
                }
                for diff in xDiffs {
                    XCTAssert(diff < 0.00001)
                }
            }
        }
    }
}
