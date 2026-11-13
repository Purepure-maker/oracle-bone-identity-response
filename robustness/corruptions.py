"""Prespecified native-image nuisance factors; no random erasure or stroke edits."""
import cv2
import numpy as np

def apply(image,condition):
    family=condition['family'];severity=condition['severity']
    if family=='clean':return image.copy()
    h,w=image.shape[:2]
    if family=='downsample':return cv2.resize(image,(max(1,round(w*severity)),max(1,round(h*severity))),interpolation=cv2.INTER_AREA)
    if family=='blur':return cv2.GaussianBlur(image,(0,0),sigmaX=severity,sigmaY=severity,borderType=cv2.BORDER_REFLECT_101)
    if family=='asymmetric_margin':
        px=max(1,round(w*severity));py=max(1,round(h*severity))
        border=np.concatenate([image[0],image[-1],image[:,0],image[:,-1]])
        result=np.empty((h+py,w+px,3),dtype=image.dtype);result[:]=np.median(border,axis=0).astype(image.dtype)
        result[py:,px:]=image
        return result
    raise ValueError('Unknown prespecified corruption family')

def self_test():
    original=np.arange(19*31*3,dtype=np.uint8).reshape(19,31,3)
    assert np.array_equal(apply(original,{'family':'clean','severity':0}),original)
    shifted=apply(original,{'family':'asymmetric_margin','severity':.2})
    assert np.array_equal(shifted[4:,6:],original)
    small=apply(original,{'family':'downsample','severity':.5});assert small.shape==(10,16,3)
    assert np.array_equal(apply(original,{'family':'blur','severity':.35}),apply(original,{'family':'blur','severity':.35}))
    return {'clean_identity':True,'margin_keeps_all_original_pixels':True,'downsample_size':True,'deterministic_blur':True}
