import SwiftUI
import CoreMotion
import TensorFlowLite
import AVFoundation


@main
struct SHLModelEvaluationApp: App {
    @StateObject private var pipeline = try! Pipeline()

    private func fmt(_ parameter: Parameter) -> String {
        String(format: "%.2f", parameter)
    }

    var content: some View {
        VStack {
            if let predictions = pipeline.predictions {
                ForEach(predictions, id: \.label.rawValue) { p -> Text in
                    Text("\(p.label.description): \(fmt(p.confidence * 100))")
                }
            }
        }
        .padding()
    }

    var body: some Scene {
        WindowGroup {
            Color.gray.overlay(content)
                .onAppear(perform: {
                    pipeline.run()

                    UIScreen.main.brightness = CGFloat(1.0)
                })
                .onDisappear(perform: pipeline.stop)
        }
    }
}
