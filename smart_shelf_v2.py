"""
this is the main file
How to run?
1) go to local paths, set paths for trained model, csv database files, video sample
2) run!
3) if you want to check user frontend, app.py needs to run, run it from command

contact info: basharw773@gmail.com
"""

import cv2
from ultralytics import YOLO
from functions import (
    shelf_boundary,
    get_ids,
    height_classification,
    describe_objects,
    update_state,
    process_all_cases,
    mov_detection_roi,
    get_heights,
    logger_init,
)

import pandas as pd
from pathlib import Path
from local_paths import trained_model_path, video_path, balance_data_path, price_data_path
model = YOLO(trained_model_path)  # load a custom model

# Open the video file
cap = cv2.VideoCapture(video_path)


# initialize parameters

counter = 0
fps = 30
init_boxes, history_list = list(), list()
initialize_end = False
bot_shelf_border, up_shelf_border = 0, 0
init_fps_count = int(fps)
state = "Initialization"
mov_detection_thresh = 100
customer_id = 125
start_items_list, items_list = list(), list()

log_updater = logger_init()
# file_handler = logging.FileHandler('logs.log')
# file_handler.setFormatter(formatter)


log_updater.info("start")

balance_data_frame = pd.read_csv(Path(balance_data_path), sep=";")
prices_data_frame = pd.read_csv(Path(price_data_path), sep=";")

## start


while cap.isOpened():
    # Read a frame from the video

    success, frame = cap.read()

    if success:
        # Break the loop if 'q' is pressed

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if counter % 10 != 0 or counter == init_fps_count:  # 5fps
            counter += 1
            if counter >= init_fps_count and initialize_end == False:
                initialize_end = True
                tracked_id_list = get_ids(init_boxes)
                bot_shelf_border, up_shelf_border, height_list = shelf_boundary(
                    tracked_id_list, init_boxes
                )
                tracked_id_list = get_ids(init_boxes)
                height_thresholds, obj_num = height_classification(height_list)
                items_list = describe_objects(height_list, height_thresholds)
                start_items_list = items_list
                print("shelf_bot ", bot_shelf_border, "shelf_top ", up_shelf_border)
                print("Items in shelf:", items_list)
                print("initialization ended\n")
        elif counter < init_fps_count:  # initialize
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # results = model(frame)

            new_state = "Initialization"
            if update_state(new_state, state):
                print(new_state)
                state = new_state
            results = model.track(
                frame,
                verbose=False,
                conf=0.8,
                # classes=39, #39 bottles,
            )
            # Tracking with default tracker

            init_boxes.append(results[0].boxes)

            last_init_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            counter += 1
        elif initialize_end == True:

            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            mov_indicator = mov_detection_roi(
                last_init_frame,
                current_frame_gray,
                bot_shelf_border,
                up_shelf_border,
                mov_detection_thresh,
            )

            last_init_frame = current_frame_gray

            results = model.track(frame, verbose=False, conf=0.8)

            ## visualize ROI

            annotated_frame = results[0].plot()
            annotated_frame_crop = annotated_frame[bot_shelf_border:up_shelf_border, :]
            resized_annotated_frame = cv2.resize(
                annotated_frame_crop, (0, 0), fx=0.4, fy=0.4
            )
            cv2.imshow("ROI", resized_annotated_frame)
            cv2.imshow("full_view", cv2.resize(annotated_frame, (0, 0), fx=0.4, fy=0.4))

            if mov_indicator:
                new_state = "customer is checking"
                if update_state(new_state, state):
                    print(new_state)
                    state = new_state
                pass
            else:

                curr_heights = get_heights(results[0].boxes)

                current_items = describe_objects(curr_heights, height_thresholds)
                history_list.append(current_items)
                final_current_lists = history_list[-5:]

                items_list, state, balance_data_frame, log_info = process_all_cases(
                    final_current_lists,
                    curr_heights,
                    items_list,
                    state,
                    new_state,
                    customer_id,
                    balance_data_frame,
                    prices_data_frame,
                    log_updater,
                )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            counter += 1
            # img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
            # img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
            # grayscale_diff = cv2.subtract(img2, last_frame)
    else:
        cap.release()
        cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()

basket = [x for x in start_items_list if (x not in items_list)]
print("\nend of view")
print("\nbasket", basket)
log_basket = "your basket" + str(basket)
log_updater.info(log_basket)
