  def random_flip(self, image):
    image = tf.image.random_flip_left_right(image)
    return image
  
  def random_pad_crop(self,image,pad,size):
    # Pad 4 pixels on each dimension of feature map, done in mini-batch
    image = tf.image.resize_image_with_crop_or_pad(image, size+2*pad, size+2*pad)
    return image  
  
  def random_crop(self, image, height, width ,depth):
    image = tf.random_crop(image, [height, width, depth])
    return image    
  
  #p : the probability that random erasing is performed
  #s_l, s_h : minimum / maximum proportion of erased area against input image
  #r_1, r_2 : minimum / maximum aspect ratio of erased area
  #v_l, v_h : minimum / maximum value for erased area
  #pixel_level : pixel-level randomization for erased area
  
  def cutout(input_image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                  v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img
      
    return eraser
      
      
