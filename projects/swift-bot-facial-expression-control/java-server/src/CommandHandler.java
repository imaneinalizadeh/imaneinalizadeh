import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * CommandHandler.java
 *
 * Parses one line of the wire protocol ("COMMAND|gaze_tag") and
 * dispatches to MotorController. Unknown commands are logged and
 * ignored rather than crashing the connection — a malformed or
 * future-version line from the Python client shouldn't take down the
 * whole server.
 */
public class CommandHandler {

    private static final Set<String> KNOWN_COMMANDS = new HashSet<>(
            Arrays.asList("ADVANCE", "RETREAT", "HOLD")
    );

    private final MotorController motorController;
    private String lastCommand = null;

    public CommandHandler(MotorController motorController) {
        this.motorController = motorController;
    }

    /**
     * Parses a line like "ADVANCE|centre" and dispatches it. Gaze tag
     * is currently logged only (not acted on) — reserved for a future
     * extension where gaze direction could steer left/right, per the
     * project's stated next-steps.
     */
    public void handleLine(String line) {
        if (line == null || line.isBlank()) {
            return;
        }

        String[] parts = line.trim().split("\\|", 2);
        String command = parts[0].trim().toUpperCase();
        String gazeTag = parts.length > 1 ? parts[1].trim() : "unknown";

        if (!KNOWN_COMMANDS.contains(command)) {
            System.err.println("Ignoring unknown command: '" + command + "' (raw line: '" + line + "')");
            return;
        }

        System.out.println("Command: " + command + "  (gaze: " + gazeTag + ")");

        if (command.equals(lastCommand)) {
            // Client already de-duplicates repeated commands before
            // sending, but the server stays defensive in case a future
            // client version doesn't.
            return;
        }

        switch (command) {
            case "ADVANCE":
                motorController.advance();
                break;
            case "RETREAT":
                motorController.retreat();
                break;
            case "HOLD":
                motorController.hold();
                break;
        }

        lastCommand = command;
    }
}
