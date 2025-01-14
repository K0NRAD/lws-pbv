import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LwsAppMQTTChat:
    def __init__(self, username):
        self.broker_address = "broker.hivemq.com"
        self.port = 1883
        self.username = username
        self.is_connected = False
        self.base_topic = "lws_app_chat"
        self.presence_topic = f"{self.base_topic}/presence"
        self.active_chat = None
        self.unread_messages = defaultdict(int)
        self.online_users = set()
        self.chat_history = defaultdict(list)

        # Client initialisieren
        client_id = f'lws-mqtt-{username}-{int(time.time())}'
        self.client = mqtt.Client(client_id=client_id, clean_session=True)

        # Callbacks setzen
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            logger.info("Verbunden mit MQTT Broker")

            # Eigenen Status-Topic und Nachrichten-Topics abonnieren
            self.client.subscribe(f"{self.presence_topic}/#")
            self.client.subscribe(f"{self.base_topic}/chat/{self.username}/#")

            # Online-Status veröffentlichen
            self.publish_presence("online")

            # Andere User über eigene Anwesenheit informieren
            self.request_online_users()
        else:
            logger.error(f"Verbindung fehlgeschlagen mit Code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            topic = msg.topic

            # Presence Updates verarbeiten
            if topic.startswith(self.presence_topic):
                username = topic.split('/')[-1]
                if username != self.username:
                    if payload == "online":
                        self.online_users.add(username)
                        if not self.active_chat:
                            self.display_menu()
                    elif payload == "offline":
                        self.online_users.discard(username)
                        if not self.active_chat:
                            self.display_menu()

            # Chat Nachrichten verarbeiten
            elif topic.startswith(f"{self.base_topic}/chat/"):
                data = json.loads(payload)
                sender = data['sender']
                message = data['message']
                timestamp = data['timestamp']

                if self.active_chat == sender:
                    print(f"\n{timestamp} {sender}: {message}")
                    print("Sie: ", end='', flush=True)
                else:
                    self.unread_messages[sender] += 1
                    self.chat_history[sender].append((timestamp, sender, message))
                    if not self.active_chat:
                        self.display_menu()

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der Nachricht: {e}")

    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        self.publish_presence("offline")
        logger.warning("Verbindung zum Broker getrennt")

    def clear_screen(self):
        """Bildschirm plattformunabhängig bereinigen"""
        try:
            # Windows
            if os.name == 'nt':
                os.system('cls')
            # Unix/Linux/MacOS
            else:
                # Versuche erst 'clear'
                if os.system('clear') != 0:
                    # Fallback: Drucke neue Zeilen
                    print('\n' * 100)
        except:
            # Fallback wenn nichts anderes funktioniert
            print('\n' * 100)

    def display_menu(self):
        self.clear_screen()
        print("\n=== LWS App Chat ===")
        print(f"Eingeloggt als: {self.username}\n")
        print("Online Benutzer:")

        for user in sorted(self.online_users):
            if user != self.username:
                unread = self.unread_messages.get(user, 0)
                status = f"({unread} ungelesen)" if unread > 0 else ""
                print(f"- {user} {status}")

        print("\nBefehle:")
        print("/chat [username] - Chat mit Benutzer starten")
        print("/exit - Programm beenden")
        print("/menu - Zurück zum Hauptmenü")

    def start_chat_with(self, target_user):
        if target_user not in self.online_users:
            print(f"Benutzer {target_user} ist nicht online.")
            return

        self.clear_screen()
        print(f"\n=== Chat mit {target_user} ===")

        # Zeige ungelesene Nachrichten
        if target_user in self.chat_history:
            for timestamp, sender, message in self.chat_history[target_user][-10:]:
                print(f"{timestamp} {sender}: {message}")

        self.unread_messages[target_user] = 0
        self.active_chat = target_user

    def publish_presence(self, status):
        self.client.publish(f"{self.presence_topic}/{self.username}", status, retain=True)

    def request_online_users(self):
        self.client.publish(f"{self.presence_topic}/request", self.username)

    def send_message(self, target_user, message):
        if not message.strip():
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        data = {
            'sender': self.username,
            'message': message,
            'timestamp': timestamp
        }

        # Nachricht an den Ziel-User senden
        self.client.publish(f"{self.base_topic}/chat/{target_user}", json.dumps(data))
        print(f"{timestamp} Sie: {message}")

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Verbindungsfehler: {e}")
            sys.exit(1)

    def disconnect(self):
        self.publish_presence("offline")
        time.sleep(0.5)  # Warten bis die Offline-Nachricht gesendet wurde
        self.client.loop_stop()
        self.client.disconnect()

    def run(self):
        self.display_menu()

        while True:
            try:
                if not self.is_connected:
                    logger.warning("Keine Verbindung. Versuche neu zu verbinden...")
                    time.sleep(5)
                    continue

                user_input = input("Sie: " if self.active_chat else "Befehl: ")

                if user_input.lower() == '/exit':
                    break

                elif user_input.lower() == '/menu':
                    self.active_chat = None
                    self.display_menu()

                elif user_input.lower().startswith('/chat '):
                    target_user = user_input[6:].strip()
                    if target_user != self.username:
                        self.start_chat_with(target_user)
                    else:
                        print("Sie können nicht mit sich selbst chatten.")

                elif self.active_chat:
                    self.send_message(self.active_chat, user_input)

                else:
                    print("Unbekannter Befehl. Verfügbare Befehle:")
                    print("/chat [username], /exit, /menu")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Fehler: {e}")


def get_username():
    while True:
        username = input("Bitte geben Sie Ihren Benutzernamen ein: ").strip()
        if username and ' ' not in username and len(username) >= 3:
            return username
        print("Der Benutzername muss mindestens 3 Zeichen lang sein und darf keine Leerzeichen enthalten.")


def main():
    parser = argparse.ArgumentParser(description='LWS App MQTT Chat')
    parser.add_argument('--debug', action='store_true', help='Debug-Modus aktivieren')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Clear screen and show welcome message
    print('\n' * 100)
    print("=== LWS App MQTT Chat ===\n")

    username = get_username()
    chat = LwsAppMQTTChat(username=username)

    try:
        chat.connect()
        chat.run()
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}", exc_info=True)
    finally:
        chat.disconnect()


if __name__ == "__main__":
    main()
