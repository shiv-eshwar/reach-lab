# Reach Lab

Double-click **Present.command** on this Mac to open the offline demonstration. Keep its Terminal window open. From any computer with Python 3.9 or newer, run `python3 present.py --open` from this folder. Study replay uses the included data and requires no installation, internet, camera or model training. Stop the server with Control-C.

[Standalone offline replay](presentation/Offline%20replay.html) opens directly in a browser with no Python setup.

[Five-slide talk with speaker notes](presentation/Reach%20Lab.pptx) · [Printable slide fallback](presentation/Reach%20Lab.pdf) · [Case comparison image](presentation/reach_examples.png)

## Four-minute rehearsal

| Time | Presenter action |
|---|---|
| 0:00–0:40 | Open the deck. Demonstrate a sideways reach and ask how two reaches with the same endpoints could differ. Introduce the 51 people and eight repetitions per person. |
| 0:40–1:40 | Switch to **Study replay**, participant **67**. Play reach 1, pause, scrub, switch repetitions and enable **All eight trails**. The gold trace follows the wrist, pale segments show the arm, and green marks the target. Path efficiency is straight-line distance divided by distance travelled. |
| 1:40–2:30 | Reveal participant 67's held-out result. Switch to **26**, reveal the result, and explain the missed DCP case. Return to the results slide. The corrected Random Forest detected 27/30 DCP and correctly classified 21/21 TD, giving 48/51 overall. |
| 2:30–3:35 | Open **Live session**. Attach a bright solid-colour marker to a wrist, select **Enable camera**, click the marker and press **Start reach**. Move sideways, then press **Finish reach**. Return the hand to the start between repetitions. Show the tracking and efficiency after one reach. A full eight-reach run enables the exploratory wrist-only score. |
| 3:35–4:00 | Show the data path from a camera to the receiver. Close with the next experiment, comparing measurements from a new camera against an independent reference and evaluating a new participant cohort. |

For a short slot, preload an eight-reach recorded device session. Run `python3 demo/send_session.py http://localhost:8765` and select **Load latest device session**. The dashboard identifies the recording as a replay. Bring a bright marker and rehearse camera permission on the actual laptop. The slides and the bundled study replay also work without camera permission.

## Connected-device demonstration

On a **trusted local network**, launch the receiver with `python3 present.py --host 0.0.0.0 --port 8765`. Open its dashboard at `http://localhost:8765`, select **Live session** and **Watch device stream**. The optional LAN receiver has no authentication. The default launcher binds to loopback only.

On the sending computer, copy this project and run the command below, substituting the receiver's LAN address. Open the sender dashboard on **localhost** so Chrome can request camera access. Select **Live session**, enable **Send captured wrist coordinates**, and record the reaches. The sender relays coordinates over HTTP. Video stays in the sending browser. The receiver can load the complete session after the eighth reach.

```sh
python3 present.py --port 8767 --relay http://RECEIVER_IP:8765 --open
```

For a repeatable network rehearsal without a camera, start **Watch device stream** and run `python3 demo/send_session.py http://localhost:8765 --stream` in another terminal. After playback, select **Load latest device session**. Use the receiver's LAN address to run the same sender on another machine. To transmit a previously exported recording, add `--file session.json`. The receiver keeps the latest stream and eight completed sessions in memory. Restarting clears them.

**Device format.** POST JSON to `/stream` with `device`, `rep` (1–8) and `path`, a list of up to 1,200 samples. POST completed sessions to `/sessions` with `device` and `reps`, a list of eight paths. Each sample is `[time_ms, wrist_x_px, wrist_y_px]` in an undistorted image plane. Timestamps increase within each reach. The camera flow rejects short movements and interrupted tracking. Tracking works best with a distinctive saturated marker, steady lighting and a stationary camera.

## Results and interpretation

| Run | Inputs | Held-out performance | Use |
|---|---|---|---|
| Corrected study Random Forest | 168 candidate descriptors from eight study reaches, including restored wrist-to-target descriptors | 48/51 correct. Three missed DCP cases, zero TD cases called DCP. | Study replay and slides. Each case's displayed score and signed contributions come from its fitted outer-fold forest. |
| Portable logistic model | Mean path efficiency, its across-reach standard deviation, and mean maximum sideways deviation divided by reach distance | 45/51 correct. Five missed DCP cases and one TD case called DCP. | Unlabeled wrist sessions. The final portable model is trained on the 51 study participants. Live-camera scores are exploratory. |
| Earlier archived Random Forest benchmark | 162 descriptors with endpoint-distance features absent | 46/51 correct | Historical comparison in the original project review. |

Both new models use participant-level evaluation. The forest benchmark uses five outer folds and four inner folds for hyperparameter selection. The portable model uses a fixed specification, with scaling fitted on the training participants. Model design was informed by this cohort. The three leading forest contributions are average changes in class probability along tree decision paths, expressed in percentage points. They describe the model's calculation. A diagnosis, severity grade or treatment decision requires separate clinical validation.

Study coordinates were recorded at 25 Hz and the source study estimated about 2 mm per pixel. Live measurements use scale-independent ratios. A coloured marker, a different camera or frame rate and manual start/finish times can change measurements. Interpolated study landmarks also affect the measurements. A missing target is shown as unavailable. The study's original videos are not required for the landmark replay.

## Rebuild

The delivered app includes its data and fitted portable model in `demo/study.json`. With **uv** installed, run `./rebuild.sh` to rebuild from the 408 published workbooks and verify all 51 stored Random Forest probabilities against the supplied benchmark. Run `./rebuild.sh --benchmark` to additionally rerun the full 16-model search on the corrected features. These commands regenerate the relevant processed and demonstration outputs. The research environment uses Python 3.11 and the pinned packages in `requirements.txt`. The presentation server uses only the Python standard library.

- `data/raw/` holds the original workbooks. The loader maps the publisher's filename typo for participant 60 and verifies complete participant/repetition coverage.
- `data/processed` points to the top-level `processed/` directory. Preserve this link when copying the research project. For the presentation alone, copy `demo/` and `present.py`.
- `results/corrected_benchmark/` contains fresh predictions, fold settings and model comparisons. Other research tables in `results/` and the case comparison PNG are historical artifacts.
- `demo/build.py` exports the replay, cohort split IDs, feature scaling, coefficients and input hash. `demo/forest.py` verifies the held-out predictions and decomposes them for display.

## Data attribution

Data adapted from Helga Haberfehlner, Inti Vanmechelen, Ellen Van Wonterghem, Jean-Marie Aerts, Hans Hallez and Elegast Monbaliu, KU Leuven Research Data Repository, version 2.1. Original workbook values were processed to create movement features, visualizations and model outputs. [Dataset](https://doi.org/10.48804/BZM9SJ) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
