# Vendored from https://github.com/sharathadavanne/seld-dcase2023 (train_seldnet.py)
# commit 24d14c9ebe6878062dc3b68eb07dd0100722f530
# MIT License, (c) Sharath Adavanne / DCASE 2023 baseline authors
# Modified for second-ear: only the multi-ACCDOA decoding of test_epoch() is kept; the per-frame
# unify logic is moved into multi_accdoa_to_output_dict() unchanged in behaviour. See README.md.
# ruff: noqa
# fmt: off
import numpy as np

from .SELD_evaluation_metrics import distance_between_cartesian_coordinates


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def multi_accdoa_to_output_dict(output, nb_classes, thresh_unify):
    """Upstream test_epoch() multi-ACCDOA branch for one file.

    Args:
        output: [frames, 3*3*nb_classes] network output (batch already flattened to frames)
    Return:
        output_dict: {frame: [[class, x, y, z], ...]} (Cartesian, same layout as upstream)
    """
    sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(output[None], nb_classes)
    sed_pred0, doa_pred0 = sed_pred0[0], doa_pred0[0]
    sed_pred1, doa_pred1 = sed_pred1[0], doa_pred1[0]
    sed_pred2, doa_pred2 = sed_pred2[0], doa_pred2[0]
    n = nb_classes
    output_dict = {}
    for frame_cnt in range(sed_pred0.shape[0]):
        for class_cnt in range(sed_pred0.shape[1]):
            # determine whether track0 is similar to track1
            flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, thresh_unify, n)
            flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, thresh_unify, n)
            flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, thresh_unify, n)
            # unify or not unify according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                for sed_p, doa_p in ((sed_pred0, doa_pred0), (sed_pred1, doa_pred1), (sed_pred2, doa_pred2)):
                    if sed_p[frame_cnt][class_cnt] > 0.5:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt, doa_p[frame_cnt][class_cnt], doa_p[frame_cnt][class_cnt+n], doa_p[frame_cnt][class_cnt+2*n]])
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if flag_0sim1:
                    lone_sed, lone_doa, a, b = sed_pred2, doa_pred2, doa_pred0, doa_pred1
                elif flag_1sim2:
                    lone_sed, lone_doa, a, b = sed_pred0, doa_pred0, doa_pred1, doa_pred2
                else:  # flag_2sim0
                    lone_sed, lone_doa, a, b = sed_pred1, doa_pred1, doa_pred2, doa_pred0
                if lone_sed[frame_cnt][class_cnt] > 0.5:
                    output_dict[frame_cnt].append([class_cnt, lone_doa[frame_cnt][class_cnt], lone_doa[frame_cnt][class_cnt+n], lone_doa[frame_cnt][class_cnt+2*n]])
                doa_pred_fc = (a[frame_cnt] + b[frame_cnt]) / 2
                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+n], doa_pred_fc[class_cnt+2*n]])
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+n], doa_pred_fc[class_cnt+2*n]])
    return output_dict
