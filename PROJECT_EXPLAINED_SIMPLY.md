# Helping a Self-Driving Car Remember Hidden Objects

## The idea

Self-driving cars use cameras and AI to find objects such as cars, trucks, bicycles, and pedestrians.

Most basic detection models examine one image at a time. If a truck briefly blocks a pedestrian, the model may act as if the pedestrian disappeared. When the pedestrian becomes visible again, the model may treat them as a completely new person.

This project tests whether an AI system can use recent images to remember that the hidden object probably still exists.

## A simple analogy

Imagine watching someone walk behind a parked bus. You do not assume that they vanished. You remember:

- What they looked like.
- Where they were moving.
- How long they have been hidden.
- Where they are likely to appear again.

Our system tries to give a camera a similar kind of short-term memory.

## How the system works

We call the proposed method **OATM**, which stands for **Occlusion-Adaptive Temporal Memory**.

1. A pretrained detector, such as YOLOX, finds visible cars and pedestrians.
2. The system records what each object looked like and how it was moving.
3. If an object disappears behind something, the system decides whether it is probably hidden or has actually left the camera view.
4. While the object is hidden, the system predicts its likely position.
5. Its confidence slowly decreases the longer the object stays hidden.
6. When the object reappears, the system tries to reconnect it to the same identity.
7. If there is not enough evidence, the system removes the prediction so it does not create an imaginary "ghost object."

The system only uses the current and previous images. It does not look at future images, because a real car would not have access to the future.

## Strategies we will compare

All strategies will use the same object detector. This makes the comparison fair: the main difference is how each strategy uses time and memory.

| Strategy | How it works | Main weakness |
|---|---|---|
| Single-frame detection | Looks at only the current image | A hidden object immediately disappears |
| SORT | Predicts movement using a simple motion pattern | Can struggle when objects turn or the camera moves |
| ByteTrack | Connects strong and weak detections between images | Still depends on seeing at least part of the object |
| Fixed temporal memory | Keeps missing objects for a fixed number of images | May forget too early or keep false objects too long |
| Proposed OATM | Uses appearance, movement, occlusion clues, and changing confidence | More complicated and must be tested carefully |

## Why we expect OATM to perform better

This is our hypothesis. The experiment will determine whether it is correct.

| OATM feature | Why it may help |
|---|---|
| Remembers the last clear appearance | Helps recognize the same object after it returns |
| Remembers movement | Helps estimate where the hidden object is going |
| Checks for an object blocking the view | Helps separate "hidden" from "gone" |
| Accounts for camera movement | Prevents the car's own motion from confusing the prediction |
| Confidence decreases over time | Stops uncertain predictions from lasting forever |
| Detects when an object leaves the image | Reduces ghost objects |

We expect OATM to keep more correct objects during short occlusions and reconnect more objects after they reappear. However, remembering objects for too long could increase false predictions. A successful method must improve memory without creating too many ghost objects.

## How we will test it

We will use camera sequences from the **nuScenes** autonomous-driving dataset.

The tests will include:

- Natural cases where one road user blocks another.
- Controlled cases where we add a realistic visual blocker for a known amount of time.
- Different object types and occlusion lengths.
- Separate scenes for training and testing.

We will measure:

- How many hidden objects remain correctly tracked.
- Whether the same identity is restored after reappearance.
- How accurately the hidden position is predicted.
- How many ghost objects are created.
- How quickly the system runs.

## How this differs from a real self-driving car

This is a focused research experiment, not a complete self-driving system.

| Our project | Real autonomous vehicle |
|---|---|
| Starts with one front camera | Usually combines several cameras, radar, and LiDAR |
| Uses recorded nuScenes sequences | Must process live sensor data immediately |
| Focuses on selected objects | Must handle many road objects and unusual situations |
| Tests controlled occlusions | Encounters unpredictable weather, lighting, and traffic |
| Measures one perception problem | Must also plan, steer, brake, and meet strict safety rules |

The goal is to answer one clear question: **Can short-term visual memory help a camera keep track of temporarily hidden road users without creating too many false predictions?**
