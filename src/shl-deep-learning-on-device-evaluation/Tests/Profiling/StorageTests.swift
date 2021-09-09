import XCTest

@testable import SHLModelEvaluation

fileprivate extension URL {
    var attributes: [FileAttributeKey : Any]? {
        do {
            return try FileManager.default.attributesOfItem(atPath: path)
        } catch let error as NSError {
            print("FileAttribute error: \(error)")
        }
        return nil
    }

    var fileSize: UInt64 {
        return attributes?[.size] as? UInt64 ?? UInt64(0)
    }
}

class StorageTests: XCTestCase {
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

    func testModelSizes() throws {
        for modelId in modelIds {
            guard
                let url = Bundle.main.url(forResource: modelId, withExtension: "tflite")
            else { XCTFail(); return }
            print("Size for model \(modelId) in MB: \(Double(url.fileSize) / 1_000_000)")
        }
    }
}
