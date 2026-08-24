import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * RobotServer.java
 *
 * TCP server that accepts a connection from the Python vision client
 * (robot_client.py), reads newline-terminated commands of the form
 * "COMMAND|gaze_tag", and dispatches each to CommandHandler.
 *
 * See docs/protocol.md for the wire format.
 *
 * Run: javac *.java && java RobotServer
 */
public class RobotServer {

    private static final int DEFAULT_PORT = 5050;

    public static void main(String[] args) {
        int port = DEFAULT_PORT;
        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }

        CommandHandler handler = new CommandHandler(new MotorController());

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("RobotServer listening on port " + port + " ...");

            while (true) {
                Socket client = serverSocket.accept();
                System.out.println("Client connected: " + client.getRemoteSocketAddress());
                handleClient(client, handler);
            }
        } catch (IOException e) {
            System.err.println("Server error: " + e.getMessage());
        }
    }

    private static void handleClient(Socket client, CommandHandler handler) {
        try (BufferedReader in = new BufferedReader(
                new InputStreamReader(client.getInputStream()))) {

            String line;
            while ((line = in.readLine()) != null) {
                handler.handleLine(line);
            }
        } catch (IOException e) {
            System.err.println("Client connection error: " + e.getMessage());
        } finally {
            try {
                client.close();
            } catch (IOException ignored) {
            }
            System.out.println("Client disconnected.");
        }
    }
}
