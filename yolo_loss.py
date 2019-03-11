from keras import backend as K
from utils import iou



def loss(y_true, y_pred, anchors):


	xy_pred = K.sigmoid(y_pred[...,1:3])
	wh_pred = anchors * K.exp(y_pred[...,3:5])
	object_pred = K.sigmoid(y_pred[...,0])
	labels_pred = K.softmax(y_pred[...,5:])


	xy_true = y_true[...,1:3]
	wh_true = y_true[...,3:5]
	object_true = y_true[..., 0]
	labels_true = y_true[...,5:]

	pred_min = xy_pred - wh_pred / 2
	pred_max = xy_pred + wh_pred / 2

	true_min = xy_true - wh_true / 2
	true_max = xy_true + wh_true / 2

	pred_box = [pred_min, pred_max]
	true_box = [true_min, true_max]

	IOU = iou(true_box, pred_box)