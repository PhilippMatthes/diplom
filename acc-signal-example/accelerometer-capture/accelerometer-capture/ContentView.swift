import SwiftUI
import CoreMotion

struct ContentView: View {
    struct Datapoint: Codable {
        let t: TimeInterval
        let v: Double
    }

    @State var accDataTX = [Datapoint]()
    @State var accDataTY = [Datapoint]()
    @State var accDataTZ = [Datapoint]()

    let mManager = CMMotionManager()

    func runAcc() {
        let accelerometerMin: Double = 0.01
        let delta: TimeInterval = 0.005
        let updateInterval = accelerometerMin + delta
        guard mManager.isAccelerometerAvailable else { fatalError() }
        mManager.accelerometerUpdateInterval = updateInterval
        let startDate = Date()
        mManager.startAccelerometerUpdates(to: .main, withHandler: { data, err  in
            guard err == nil, let data = data else { fatalError() }
            let diff = startDate.distance(to: Date())
            accDataTX.append(Datapoint(t: diff, v: data.acceleration.x))
            accDataTY.append(Datapoint(t: diff, v: data.acceleration.y))
            accDataTZ.append(Datapoint(t: diff, v: data.acceleration.z))
        })
    }

    func stopAcc() {
        mManager.stopAccelerometerUpdates()
        print("XVALUES")
        print(String(data: try! JSONEncoder().encode(accDataTX), encoding: .utf8)!)
        print("YVALUES")
        print(String(data: try! JSONEncoder().encode(accDataTY), encoding: .utf8)!)
        print("ZVALUES")
        print(String(data: try! JSONEncoder().encode(accDataTZ), encoding: .utf8)!)
    }

    var body: some View {
        Button("Run", action: runAcc).padding()
        Button("Stop", action: stopAcc).padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
