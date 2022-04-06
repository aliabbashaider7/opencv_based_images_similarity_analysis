import cv2

class CompareImages(object):

    def __init__(self, image_1, image_2):
        self.minimum_commutative_image_diff = 1
        self.image_1 = image_1
        self.image_2 = image_2

    def compare_image(self):
        commutative_image_diff = self.get_image_difference(self.image_1, self.image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            return commutative_image_diff

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff
