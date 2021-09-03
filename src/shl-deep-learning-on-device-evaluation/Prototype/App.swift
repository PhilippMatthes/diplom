import SwiftUI
import CoreMotion
import TensorFlowLite
import AVFoundation

@main
struct SHLModelEvaluationApp: App {
    @StateObject private var pipeline = try! Pipeline()

    var content: some View {
        ScrollView {
            VStack {
                if let predictions = pipeline.predictions {
                    LabelsView(predictions: predictions)
                        .frame(height: 512)
                }
                ForEach(Sensor.order, id: \.rawValue) { sensor in
                    if let values = pipeline.sensorWindows[sensor]?.values {
                        WindowView(sensor: sensor, window: values)
                            .frame(height: 128)
                    } else {
                        EmptyView()
                    }
                }
            }
            .padding()
        }
    }

    var body: some Scene {
        WindowGroup {
            content
                .onAppear(perform: {
                    pipeline.run()

                    UIScreen.main.brightness = CGFloat(1.0)
                })
                .onDisappear(perform: pipeline.stop)
        }
    }
}
