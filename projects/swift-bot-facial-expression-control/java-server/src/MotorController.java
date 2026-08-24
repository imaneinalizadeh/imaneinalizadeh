/**
 * MotorController.java
 *
 * Abstracts the actual Swift Bot motor/actuator interface. This
 * implementation logs actions and tracks a simple position value —
 * swap the method bodies for real GPIO/serial calls to the physical
 * robot without touching CommandHandler or RobotServer.
 */
public class MotorController {

    private int position = 0; // arbitrary units; +1 per ADVANCE, -1 per RETREAT

    public void advance() {
        position += 1;
        System.out.println("  -> Motors: ADVANCE (position=" + position + ")");
        // TODO: replace with real motor driver call, e.g.
        //   serialPort.write("MOVE_FORWARD\n");
    }

    public void retreat() {
        position -= 1;
        System.out.println("  -> Motors: RETREAT (position=" + position + ")");
        // TODO: replace with real motor driver call, e.g.
        //   serialPort.write("MOVE_BACKWARD\n");
    }

    public void hold() {
        System.out.println("  -> Motors: HOLD (position=" + position + ")");
        // TODO: replace with real motor driver call, e.g.
        //   serialPort.write("STOP\n");
    }

    public int getPosition() {
        return position;
    }
}
