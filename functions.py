# -*- coding: utf-8 -*-

"""
Created on Sat May 25 10:11:25 2024

@author: ASUS
"""

import numpy as np
import logging
import pandas as pd


def get_price(prices_data_frame: pd.DataFrame(), product_name="small bottle") -> int:
    product_price = prices_data_frame[prices_data_frame["product"] == product_name][
        "price"
    ].iloc[0]
    return product_price


def update_balance(balance_data_frame: pd.DataFrame(), customer_id, amount=0):

    new_balance = (
        balance_data_frame[balance_data_frame["person_id"] == customer_id].balance
        + amount
    )
    balance_data_frame[balance_data_frame["person_id"] == customer_id].iloc[
        0
    ].balance = new_balance
    balance_data_frame.loc[
        balance_data_frame["person_id"] == customer_id, "balance"
    ] = new_balance

    log_info = f"{amount} was added tu your balance"
    return balance_data_frame, log_info


def logger_init():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler("logs.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# detect move in region of interest 
def mov_detection_roi(
    last_init_frame, current_frame_gray, bot_shelf_border, up_shelf_border, thresh
)->bool:
    last_init_frame_crop = last_init_frame[bot_shelf_border:up_shelf_border, :]
    current_frame_shelf = current_frame_gray[bot_shelf_border:up_shelf_border, :]

    diff = abs(
        np.array(current_frame_shelf, dtype=float)
        - np.array(last_init_frame_crop, dtype=float)
    )

    if np.max(diff) > 100:
        return True
    else:
        return False

def process_all_cases(
    final_current_lists,
    curr_heights,
    items_list,
    state,
    new_state,
    customer_id,
    balance_data_frame,
    prices_data_frame,
    log_updater,
):
    log_info = ""
    if len(final_current_lists) < 5:
        return sorted(items_list), state, balance_data_frame, log_info
    final_items = get_final_items_list(final_current_lists)
    if len(curr_heights) < len(items_list):
        items_list, state, balance_data_frame, log_info = update_after_taking(
            items_list,
            final_items,
            state,
            new_state,
            customer_id,
            balance_data_frame,
            prices_data_frame,
            log_updater,
        )

    elif len(curr_heights) > len(items_list):
        items_list, state, balance_data_frame, log_info = update_after_returning(
            items_list,
            final_items,
            state,
            new_state,
            customer_id,
            balance_data_frame,
            prices_data_frame,
            log_updater,
        )
        
    # multiple events action, return and take
    elif sorted(items_list) != sorted(final_items):
            items_list, state, balance_data_frame, log_info = update_after_returning(
                items_list,
                final_items,
                state,
                new_state,
                customer_id,
                balance_data_frame,
                prices_data_frame,
                log_updater,
            )
            items_list, state, balance_data_frame, log_info = update_after_taking(
                items_list,
                final_items,
                state,
                new_state,
                customer_id,
                balance_data_frame,
                prices_data_frame,
                log_updater,
            )
    return sorted(items_list), state, balance_data_frame, log_info


def get_heights(result_boxes)->list:
    curr_heights = list()
    for object_box in result_boxes:
        for ii in range(0, len(object_box.id)):
            current_obj_boundary = object_box.xywh.numpy()[ii]
            curr_heights.append(current_obj_boundary[3])
    return curr_heights


def get_final_items_list(final_current_lists)->list:
    final_items = list()
    for final_item in final_current_lists:
        for ii in final_item:
            final_items.append(ii)
    final_items = sorted(list(np.unique(final_items)))
    return final_items


def update_after_returning(
    init_items: list,
    current_items: list,
    state: bool,
    new_state: bool,
    customer_id: int,
    balance_data_frame,
    prices_data_frame,
    log_updater,
):
    log_info = ""
    return_list = [x for x in current_items if (x not in init_items)]
    if return_list:
        for returned_item in return_list:
            new_state = str("The customer returns the " + returned_item)
            price = get_price(prices_data_frame, product_name=returned_item)
            balance_data_frame, log_info = update_balance(
                balance_data_frame, customer_id, price
            )
            init_items.append(returned_item)
            if update_state(new_state, state):
                print(new_state)
                state = new_state

                log_info = (
                    "you have returned "
                    + returned_item
                    + " "
                    + str(price)
                    + " was added to your balance"
                )
                log_updater.info(log_info)
    return sorted(init_items), state, balance_data_frame, log_info


def update_after_taking(
    init_items: list,
    current_items: list,
    state: bool,
    new_state: bool,
    customer_id,
    balance_data_frame,
    prices_data_frame,
    log_updater,
):
    log_info = ""
    buy_factor = -1
    gone_list = [x for x in init_items if (x not in current_items)]
    if gone_list:
        for taken_item in gone_list:
            new_state = str("The customer takes the " + taken_item)
            price = get_price(prices_data_frame, product_name=taken_item)
            balance_data_frame, log_info = update_balance(
                balance_data_frame, customer_id, buy_factor * price
            )
            init_items.remove(taken_item)
            if update_state(new_state, state):
                print(new_state)
                state = new_state
                log_info = (
                    "you have bought "
                    + taken_item
                    + " "
                    + str(price)
                    + " was discounted from your balance"
                )
                log_updater.info(log_info)
    return sorted(init_items), state, balance_data_frame, log_info


# get objects in initial state
def get_ids(init_boxes: list)->list:
    tracked_id_list = list([])
    for object_box in init_boxes:
        if object_box.id != None:
            for ii in object_box.id.numpy():
                tracked_id_list.append(ii)
    tracked_id_list = list(np.unique(tracked_id_list))
    return tracked_id_list


def shelf_boundary(tracked_id_list, init_boxes):
    tracked_boxes = list([])

    tofind_list = tracked_id_list
    for object_box in init_boxes:
        for ii in range(0, len(object_box.id)):
            if object_box.id[ii].numpy() in tofind_list:
                tracked_boxes.append(object_box.xywh.numpy()[ii])
                tofind_list.remove(object_box.id[ii].numpy())
    obj_count = len(tracked_boxes)
    x_list, y_list, width_list, height_list = list(), list(), list(), list()
    for ii in range(0, obj_count):
        x_list.append(tracked_boxes[ii][0])
        y_list.append(tracked_boxes[ii][1])
        width_list.append(tracked_boxes[ii][2])
        height_list.append(tracked_boxes[ii][3])
    bot_shelf_border = int(min(np.array(y_list)) - int(0.5 * max(height_list)))
    up_shelf_border = int(max(np.array(y_list)) + int(0.5 * max(height_list)))
    return bot_shelf_border, up_shelf_border, height_list


def update_state(newstate: str, state: str):
    if newstate != state:
        return True
    else:
        return False

def describe_objects(height_list, height_thresholds):
    item_list = list()
    for height in height_list:
        if height < height_thresholds[0]:
            item_list.append("small bottle")
        elif height > height_thresholds[len(height_thresholds) - 1]:
            item_list.append("large bottle")
        else:
            item_list.append("medium bottle")
    return sorted(item_list)


def height_classification(height_list: list):
    height_thresholds = list([])
    if len(height_list) == 2:
        height_thresholds.append(int(np.mean([height_list[0], height_list[1]])))
    elif len(height_list) == 3:
        height_thresholds.append(int(np.mean([height_list[0], height_list[1]])))
        height_thresholds.append(int(np.mean([height_list[1], height_list[2]])))
    else:
        raise ValueError("only one, or more than 3 objects detected !!")
    return height_thresholds, len(height_thresholds)
