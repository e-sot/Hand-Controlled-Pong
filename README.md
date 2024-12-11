# Pong Game with Hand Tracking
A modern take on the classic Pong game using computer vision and hand tracking technology. Players can control their paddles using hand gestures captured through a webcam.

## Features

    Real-time hand tracking for paddle control
    AI-controlled paddles when no hands are detected
    Dynamic ball physics with speed adjustments
    Combo system with power levels
    Particle effects and ball trails
    Score tracking and game over screen
    Webcam feed display in corner

## Technical Requirements

    Python 3.x
    OpenCV (cv2)
    NumPy
    MediaPipe (for hand tracking)

## Game Controls

    Move your hands in front of the webcam to control the paddles
    Press 'R' to reset the game
    Press 'Q' to quit

## Game Mechanics

    Each player controls a paddle on either side of the screen
    Score points by getting the ball past your opponent's paddle
    Build combos by hitting the ball multiple times in quick succession
    Power levels increase with higher combos
    First player to reach 10 points wins

## Installation

    Clone the repository
    Create a virtual environment
    Install the required dependencies
    Run script.py to start the game

## Features in Detail
Paddle Control

    Left hand controls the left paddle
    Right hand controls the right paddle
    AI takes over when no hands are detected

## Ball Physics

    Dynamic speed adjustments based on combos
    Particle effects on collisions
    Visual trail effect behind the ball

## Scoring System

    Real-time score display
    Combo counter
    Power level indicator
    Game over screen with final results
