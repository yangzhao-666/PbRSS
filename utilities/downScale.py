from skimage.transform import resize

def downScale(state):
    return 255 * resize(state, (80, 80, 3))
