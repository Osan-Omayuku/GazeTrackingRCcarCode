void setup() {
    Serial.begin(9600);  // Set baud rate to match Python's serial connection
    Serial.println("Ready to receive data...");
}

void loop() {
    if (Serial.available() > 0) {
        String direction = Serial.readStringUntil('\n');  // Read the string sent from Python

        // Perform actions based on the direction
        if (direction == "left") {
            // Action for left (e.g., light up an LED or print to Serial Monitor)
            Serial.println("Moving Left");
        } else if (direction == "right") {
            // Action for right
            Serial.println("Moving Right");
        } else if (direction == "up") {
            // Action for up
            Serial.println("Moving Up");
        } else if (direction == "down") {
            // Action for down
            Serial.println("Moving Down");
        } else if (direction == "center") {
            // Action for center
            Serial.println("Center Position");
        }
    }
}
