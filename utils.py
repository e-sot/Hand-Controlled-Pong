import cv2
import mediapipe as mp
import numpy as np


class HandDetector():
    def __init__(self, mode=False, max_Hands=2, mcomplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.max_Hands = max_Hands
        self.mcomplexity = mcomplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Configuration améliorée de MediaPipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=max_Hands,
            model_complexity=mcomplexity,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Paramètres de lissage améliorés
        self.previousHandPositions = {}
        self.handVelocities = {}
        self.smoothingFactor = 0.7
        self.positionHistory = {}
        self.historyLength = 5

    def findHands(self, img, draw=True, enhancedVisuals=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    if enhancedVisuals:
                        self.mpDraw.draw_landmarks(
                            img, handlms,
                            self.mpHands.HAND_CONNECTIONS,
                            self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                            self.mpDraw.DrawingSpec(color=(255, 255, 0), thickness=2)
                        )
                    else:
                        self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, flipType=False):
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []

        if self.results.multi_hand_landmarks:
            for handType, handLMS in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []

                for id, lm in enumerate(handLMS.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                # Calcul de la boîte englobante avec marge adaptative
                margin = int(0.1 * (max(xList) - min(xList)))
                xmin, xmax = min(xList) - margin, max(xList) + margin
                ymin, ymax = min(yList) - margin, max(yList) + margin
                bbox = max(0, xmin), max(0, ymin), min(w - xmin, xmax - xmin), min(h - ymin, ymax - ymin)

                # Calcul du centre avec lissage
                center = (xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2)

                # Correction de la détection gauche/droite
                handID = handType.classification[0].label
                actualType = handType.classification[0].label

                # Gestion de l'historique des positions
                if handID not in self.positionHistory:
                    self.positionHistory[handID] = []

                self.positionHistory[handID].append(center)
                if len(self.positionHistory[handID]) > self.historyLength:
                    self.positionHistory[handID].pop(0)

                # Lissage de la position
                if len(self.positionHistory[handID]) > 0:
                    weights = np.linspace(0.5, 1.0, len(self.positionHistory[handID]))
                    weights = weights / np.sum(weights)
                    smoothed_x = int(np.sum([p[0] * w for p, w in zip(self.positionHistory[handID], weights)]))
                    smoothed_y = int(np.sum([p[1] * w for p, w in zip(self.positionHistory[handID], weights)]))
                    center = (smoothed_x, smoothed_y)

                # Calcul de la vélocité
                if handID in self.previousHandPositions:
                    prev_center = self.previousHandPositions[handID]
                    velocity = (
                        center[0] - prev_center[0],
                        center[1] - prev_center[1]
                    )
                    if handID in self.handVelocities:
                        self.handVelocities[handID] = (
                            self.smoothingFactor * velocity[0] + (1 - self.smoothingFactor) *
                            self.handVelocities[handID][0],
                            self.smoothingFactor * velocity[1] + (1 - self.smoothingFactor) *
                            self.handVelocities[handID][1]
                        )
                    else:
                        self.handVelocities[handID] = velocity

                self.previousHandPositions[handID] = center

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = center
                if handID in self.handVelocities:
                    myHand["velocity"] = self.handVelocities[handID]

                myHand["type"] = actualType
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLMS,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

        return img, allHands


def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape

    # Conversion des positions en entiers
    x1 = max(int(pos[0]), 0)
    y1 = max(int(pos[1]), 0)
    x2 = min(int(pos[0] + wf), wb)
    y2 = min(int(pos[1] + hf), hb)
    x1_overlay = 0 if pos[0] >= 0 else int(-pos[0])
    y1_overlay = 0 if pos[1] >= 0 else int(-pos[1])

    # Dimensions de la zone de superposition
    wf = int(x2 - x1)
    hf = int(y2 - y1)

    if wf <= 0 or hf <= 0:
        return imgBack

    # Extraction et application du canal alpha avec anti-aliasing
    try:
        alpha = cv2.GaussianBlur(
            imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3].astype(float) / 255.0,
            (3, 3), 0
        )
        inv_alpha = 1.0 - alpha

        # Superposition avec anti-aliasing
        for c in range(0, 3):
            imgBack[y1:y2, x1:x2, c] = (
                    imgBack[y1:y2, x1:x2, c] * inv_alpha +
                    imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, c] * alpha
            )
    except Exception as e:
        print(f"Error in overlayPNG: {e}")
        return imgBack

    return imgBack
