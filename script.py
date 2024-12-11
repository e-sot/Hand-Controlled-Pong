import cv2
import utils as ht
import numpy as np
import time
import os
import logging
import absl.logging

# Configuration pour supprimer les avertissements
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"


class PongGame:
    def __init__(self):
        # Configuration de base
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.handTracking = ht.HandDetector(detectionCon=0.7)

        # Chargement des images
        self.imageFrontDesign = cv2.imread("Resources/bck_grind.png")
        self.imageGameOver = cv2.imread("Resources/bck_grind.png")
        self.imageBat1 = cv2.imread("Resources/paddle_1.png", cv2.IMREAD_UNCHANGED)
        self.imageBat2 = cv2.imread("Resources/paddle_2.png", cv2.IMREAD_UNCHANGED)
        self.imageBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)

        # Configuration des palets et limites
        self.ai_paddle_speed = 25
        self.ai_left_y = 360
        self.ai_right_y = 360
        self.winning_score = 10
        self.paddle_height = 150
        self.paddle_top_limit = 20
        self.paddle_bottom_limit = 550
        self.current_hands = []
        self.ai_momentum = {'left': 0, 'right': 0}
        self.active_paddles = {'left': None, 'right': None}

        # Initialisation du jeu
        self.resetGame()

    def resetGame(self):
        self.ball_position = [640, 360]
        self.base_speed = 15
        self.speedX = self.base_speed
        self.speedY = self.base_speed
        self.score = [0, 0]
        self.gameOver = False
        self.combo = 0
        self.power_level = 1
        self.ball_trail = []
        self.max_trail_length = 5
        self.last_hit_time = time.time()
        self.particle_effects = []
        self.ai_left_y = 360
        self.ai_right_y = 360
        self.ai_momentum = {'left': 0, 'right': 0}
        self.active_paddles = {'left': None, 'right': None}

    def addParticleEffect(self, x, y):
        num_particles = 10
        for _ in range(num_particles):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 5)
            lifetime = 20
            self.particle_effects.append({
                'x': x,
                'y': y,
                'dx': speed * np.cos(angle),
                'dy': speed * np.sin(angle),
                'lifetime': lifetime
            })

    def updateParticles(self):
        updated_particles = []
        for particle in self.particle_effects:
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            particle['lifetime'] -= 1
            if particle['lifetime'] > 0:
                updated_particles.append(particle)
        self.particle_effects = updated_particles

    def drawParticles(self, frame):
        for particle in self.particle_effects:
            pos = (int(particle['x']), int(particle['y']))
            alpha = particle['lifetime'] / 20
            cv2.circle(frame, pos, 2, (255, 255, 0), -1)

    def updateBallTrail(self):
        self.ball_trail.append(self.ball_position.copy())
        if len(self.ball_trail) > self.max_trail_length:
            self.ball_trail.pop(0)

    def drawBallTrail(self, frame):
        for i, pos in enumerate(self.ball_trail):
            alpha = (i + 1) / len(self.ball_trail)
            size = int(20 * alpha)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), size, (0, 255, 255), 1)

    def update_ai_paddles(self):
        momentum_factor = 0.8
        max_momentum = 15

        future_ball_pos = [
            self.ball_position[0] + self.speedX * 2,
            self.ball_position[1] + self.speedY * 2
        ]

        target_y = future_ball_pos[1] - self.paddle_height / 2

        # Palet gauche (IA)
        if not self.active_paddles['left']:
            if self.ball_position[0] < 640:
                distance = target_y - self.ai_left_y
                self.ai_momentum['left'] = (self.ai_momentum['left'] * momentum_factor +
                                            np.sign(distance) * min(abs(distance) * 0.1, self.ai_paddle_speed))
                self.ai_momentum['left'] = np.clip(self.ai_momentum['left'], -max_momentum, max_momentum)
                self.ai_left_y += self.ai_momentum['left']
            else:
                self.ai_momentum['left'] *= 0.95
            self.ai_left_y = np.clip(self.ai_left_y, self.paddle_top_limit, self.paddle_bottom_limit)

        # Palet droit (IA)
        if not self.active_paddles['right']:
            if self.ball_position[0] > 640:
                distance = target_y - self.ai_right_y
                self.ai_momentum['right'] = (self.ai_momentum['right'] * momentum_factor +
                                             np.sign(distance) * min(abs(distance) * 0.1, self.ai_paddle_speed))
                self.ai_momentum['right'] = np.clip(self.ai_momentum['right'], -max_momentum, max_momentum)
                self.ai_right_y += self.ai_momentum['right']
            else:
                self.ai_momentum['right'] *= 0.95
            self.ai_right_y = np.clip(self.ai_right_y, self.paddle_top_limit, self.paddle_bottom_limit)

    def handleHit(self, is_left, x_pos, y_pos):
        current_time = time.time()
        if current_time - self.last_hit_time < 0.5:
            self.combo += 1
            if self.combo > 3:
                self.power_level = min(3, self.combo // 3)
        else:
            self.combo = 1
            self.power_level = 1

        self.last_hit_time = current_time
        self.addParticleEffect(self.ball_position[0], self.ball_position[1])

        # Limitation de la vitesse
        speed_multiplier = min(1.1, 1 + (self.combo * 0.05))
        self.speedX = -(self.speedX * speed_multiplier)

        # Limiter la vitesse maximale
        max_speed = self.base_speed * 2
        self.speedX = np.clip(self.speedX, -max_speed, max_speed)

    def drawGameInfo(self, frame):
        cv2.putText(frame, str(self.score[0]), (288, 651), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(frame, str(self.score[1]), (988, 651), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(frame, f"Combo: x{self.combo}", (550, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Power: {self.power_level}", (550, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    def handleGameOver(self, frame):
        frame = self.imageGameOver.copy()
        winner = "Left Player" if self.score[0] >= self.winning_score else "Right Player"
        cv2.putText(frame, f"{winner} Wins!", (500, 300), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"Final Score: {self.score[0]} - {self.score[1]}", (450, 350), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"Max Combo: x{self.combo}", (500, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
           #frame = cv2.flip(frame, 1)
            framecopy = frame.copy()

            frame, hands = self.handTracking.findPosition(frame, draw=True, flipType=True)
            self.current_hands = hands
            frame = cv2.addWeighted(frame, 0.3, self.imageFrontDesign, 0.7, 0)

            # Réinitialiser les palets actifs
            self.active_paddles = {'left': None, 'right': None}

            # Mise à jour des positions IA
            self.update_ai_paddles()

            if hands:
                for hand in hands:
                    x, y, w, h = hand["bbox"]
                    h1, w1, _ = self.imageBat1.shape
                    y1 = y - h1 // 2
                    y1 = np.clip(y1, self.paddle_top_limit, self.paddle_bottom_limit)

                    if hand["type"] == "Right" and not self.active_paddles['right']:
                        self.active_paddles['right'] = y1
                        frame = ht.overlayPNG(frame, self.imageBat2, (1192, y1))
                        paddle_rect = {'x': 1162, 'y': y1, 'w': 60, 'h': h1}
                        if (paddle_rect['x'] < self.ball_position[0] < paddle_rect['x'] + paddle_rect['w'] and
                                paddle_rect['y'] < self.ball_position[1] < paddle_rect['y'] + paddle_rect['h']):
                            if self.speedX > 0:
                                self.handleHit(False, 1192, y1)
                                self.ball_position[0] = paddle_rect['x']

                    if hand["type"] == "Left" and not self.active_paddles['left']:
                        self.active_paddles['left'] = y1
                        frame = ht.overlayPNG(frame, self.imageBat1, (73, y1))
                        paddle_rect = {'x': 73, 'y': y1, 'w': 60, 'h': h1}
                        if (paddle_rect['x'] < self.ball_position[0] < paddle_rect['x'] + paddle_rect['w'] and
                                paddle_rect['y'] < self.ball_position[1] < paddle_rect['y'] + paddle_rect['h']):
                            if self.speedX < 0:
                                self.handleHit(True, 73, y1)
                                self.ball_position[0] = paddle_rect['x'] + paddle_rect['w']

            # Gestion des palets IA
            if not self.active_paddles['left']:
                frame = ht.overlayPNG(frame, self.imageBat1, (73, int(self.ai_left_y)))
                paddle_rect = {'x': 73, 'y': self.ai_left_y, 'w': 60, 'h': self.paddle_height}
                if (paddle_rect['x'] < self.ball_position[0] < paddle_rect['x'] + paddle_rect['w'] and
                        paddle_rect['y'] < self.ball_position[1] < paddle_rect['y'] + paddle_rect['h']):
                    if self.speedX < 0:
                        self.handleHit(True, 73, self.ai_left_y)
                        self.ball_position[0] = paddle_rect['x'] + paddle_rect['w']

            if not self.active_paddles['right']:
                frame = ht.overlayPNG(frame, self.imageBat2, (1192, int(self.ai_right_y)))
                paddle_rect = {'x': 1162, 'y': self.ai_right_y, 'w': 60, 'h': self.paddle_height}
                if (paddle_rect['x'] < self.ball_position[0] < paddle_rect['x'] + paddle_rect['w'] and
                        paddle_rect['y'] < self.ball_position[1] < paddle_rect['y'] + paddle_rect['h']):
                    if self.speedX > 0:
                        self.handleHit(False, 1192, self.ai_right_y)
                        self.ball_position[0] = paddle_rect['x']

            if not self.gameOver:
                self.ball_position[0] += self.speedX
                self.ball_position[1] += self.speedY

                # Nouvelles limites pour les rebonds verticaux
                top_boundary = 85  # Limite haute du cadre
                bottom_boundary = 635  # Limite basse du cadre

                # Collision avec les bords verticaux
                if self.ball_position[1] >= bottom_boundary:
                    self.ball_position[1] = bottom_boundary
                    self.speedY = -self.speedY
                    self.addParticleEffect(self.ball_position[0], self.ball_position[1])
                elif self.ball_position[1] <= top_boundary:
                    self.ball_position[1] = top_boundary
                    self.speedY = -self.speedY
                    self.addParticleEffect(self.ball_position[0], self.ball_position[1])

                # Limites horizontales pour le score
                if self.ball_position[0] < 16:
                    self.score[1] += 1
                    self.ball_position = [640, 360]
                    self.speedX = self.base_speed
                    self.speedY = self.base_speed
                    self.combo = 0
                elif self.ball_position[0] > 1276:
                    self.score[0] += 1
                    self.ball_position = [640, 360]
                    self.speedX = -self.base_speed
                    self.speedY = self.base_speed
                    self.combo = 0

                if max(self.score) >= self.winning_score:
                    self.gameOver = True

            self.updateBallTrail()
            self.updateParticles()
            self.drawBallTrail(frame)
            self.drawParticles(frame)
            frame = ht.overlayPNG(frame, self.imageBall, (int(self.ball_position[0]), int(self.ball_position[1])))

            self.drawGameInfo(frame)
            if self.gameOver:
                frame = self.handleGameOver(frame)

            frame[556:700, 21:190] = cv2.resize(framecopy, (169, 144))
            webcam_frame = cv2.resize(framecopy, (169, 144))
            frame[556:700, 21:190] = webcam_frame  # Superposition de la webcam en dernier

            cv2.imshow("Ping Pong Game", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('r'):
                self.resetGame()
            elif key & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = PongGame()
    game.run()