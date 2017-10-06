from skimage import io, color, filter, morphology, feature, img_as_float
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy as sp


class CoinCounter:

    def __init__(self, closing=7, opening=10):
        # Create disk size for erosion/dilation
        self.close_disk = morphology.square(closing)
        self.open_disk = morphology.square(opening)

        self.coins = {
            'penny': 0.01,
            'nickel': 0.05,
            'dime': 0.10,
            'quarter': 0.25
        }

        # Change calibration image file here:
        self.COIN_CALIBRATION_IMAGE = "images/calibration.jpg"

        # Change what type of coin each label number represents:
        self.coin_labels = {
            'quarter': 1,
            'dime': 2,
            'penny': 3,
            'nickel': 4
        }

        self.num_coins = 0
        self.max_ratio = {}
        self.min_ratio = {}

        # Perform calibration on coins.
        self.calibrate_coins()

    def load_color_grayscale_and_binary(self, image_file):
        """
        Returns color, grayscale, and binary image.
        """
        self.color_img = io.imread(image_file)
        self.gray_img = color.rgb2grey(self.color_img)
        self.threshold = filter.threshold_otsu(self.gray_img)
        self.binary_img = self.gray_img > self.threshold

    def patch_holes_and_label(self):
        """
        Apply opening and closing to remove static from binary image.
        Then apply component labeling.
        """
        # Fill holes and removes noise.
        self.binary_img = morphology.binary_closing(self.binary_img, self.close_disk)
        self.binary_img = morphology.binary_opening(self.binary_img, self.open_disk)

        # Apply componenet labeling with watershed segmentation.
        distance = ndimage.distance_transform_edt(self.binary_img)
        local_max = feature.peak_local_max(distance, labels=self.binary_img, indices=False)
        markers = ndimage.label(local_max)[0]
        self.labels = morphology.watershed(-distance, markers, mask=self.binary_img)

    def calibrate_coins(self):
        """
        Calibrates ratios for radius size for coins. This function
        should be modified for different calibrations.
        """
        # Load color, grayscale, binary images
        print "Performing coin calibration:"
        self.load_color_grayscale_and_binary(self.COIN_CALIBRATION_IMAGE)
        self.patch_holes_and_label()
        self.compute_coin_area()

    def compute_coin_area(self):
        """
        Computes the area ratio for each coin in component label
        """
        self.money_area = {}
        for coin_type in self.coin_labels:
            coin = (self.labels == self.coin_labels[coin_type])
            area = float(sp.count_nonzero(coin))
            self.money_area[coin_type] = area

        # Compute ratio with smallest coin
        self.min_ratio['dime'] = 1.0
        self.min_ratio['quarter'] = self.money_area['quarter'] / \
            self.money_area['dime']
        self.min_ratio['penny'] = self.money_area['penny'] / \
            self.money_area['dime']
        self.min_ratio['nickel'] = self.money_area['nickel'] / \
            self.money_area['dime']

        # Compute ratio with largest coin
        self.max_ratio['quarter'] = 1.0
        self.max_ratio['dime'] = self.money_area['dime'] / \
            self.money_area['quarter']
        self.max_ratio['penny'] = self.money_area['penny'] / \
            self.money_area['quarter']
        self.max_ratio['nickel'] = self.money_area['nickel'] / \
            self.money_area['quarter']
        self.print_coin_calibration()

    def compute_coin_color(self):
        """
        Computes color distance to red of each coin
        """
        # Removing coins that are reddish
        thresh_red = 0.55
        image = img_as_float(self.color_img)
        black_mask = color.rgb2gray(image) < 0.25
        distance_red = color.rgb2gray(1 - sp.absolute(image - (1, 0, 0)))
        distance_red[black_mask] = 0
        self.gray_img = distance_red > thresh_red

    def count_money(self, image_file, num_bills):
        """
        Sets total money to zero and starts to count money.
        Coins are counted with penny color highting and ratio with
        smallest and largest coins. Bills are counted with edge/line
        detection and area ratio with other bills.
        """
        # Initialize values
        self.money_matches = {}
        self.money_area = []
        self.error_total = 0
        self.money_total = 0
        noise_thresh = 100
        label_num = 0
        count = 0

        # Load image and binarize
        self.load_color_grayscale_and_binary(image_file)

        # Find area of each label, eliminate noise
        self.patch_holes_and_label()
        self.show_image(self.labels)
        for coins in self.labels:
            coin = (self.labels == label_num)
            area = float(sp.count_nonzero(coin))
            if area > noise_thresh:
                self.money_area.append(area)
            label_num += 1

        # Remove background and sort area desc order
        del self.money_area[0]
        self.money_area = sorted(self.money_area, reverse=True)

        # Separate bills with coins by indices
        bill_size = max(self.money_area)
        min_size, max_size = None, None
        if num_bills < len(self.money_area):
            min_size = min(self.money_area[num_bills:])
            max_size = max(self.money_area[num_bills:])

        # Count money based on area
        for item in self.money_area:
            if count < num_bills:
                matching = self._match_bill(item / bill_size)
            else:
                matching = self._match_coin(item / min_size, item / max_size)
            # Increment count of coin or bill
            if matching in self.money_matches:
                self.money_matches[matching] += 1
            else:
                self.money_matches[matching] = 1
            count += 1

        print "\nMoney counted:"
        print self.money_matches

    def print_coin_calibration(self):
        """
        Outputs area of objects to screen
        """
        print "\nRatio with smallest coin:"
        print self.min_ratio
        print "\nRatio with largest coin:"
        print self.max_ratio

    def show_image(self, image):
        """
        Shows image with matplotlib
        """
        plt.show(io.imshow(image))

    def show_histogram_comparison(self):
        """
        Shows histogram with matplotlib
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        ax1.imshow(self.gray_img, cmap=plt.cm.gray)
        ax1.set_title('Grayscale')
        ax1.axis('off')

        ax2.hist(self.gray_img)
        ax2.set_title('Histogram of Grayscale Image with Thresh')
        ax2.axvline(self.threshold, color='r')

        ax3.imshow(self.binary_img, cmap=plt.cm.gray)
        ax3.set_title('Binary')
        ax3.axis('off')
        plt.show()

    def _match_coin(self, ratio_to_min, ratio_to_max):
        """
        Tries to determine which coin based on ratio to
        largest and smallest coin during calibration and
        distance to color red.
        """
        # Find out closest to smallest and largest coin
        min_match = self._find_in_dictionary(ratio_to_min, self.min_ratio)
        max_match = self._find_in_dictionary(ratio_to_max, self.max_ratio)
        closest_match = None

        # Coins match
        if min_match == max_match:
            closest_match = min_match
        else:  # compute error
            error_1 = abs((self.min_ratio[min_match] - ratio_to_min)) + \
                abs((self.max_ratio[min_match] - ratio_to_max))
            error_2 = abs((self.min_ratio[max_match] - ratio_to_min)) + \
                abs((self.max_ratio[max_match] - ratio_to_max))

            if error_1 < error_2:
                closest_match = min_match
                self.error_total += self.coins[max_match]
            else:
                closest_match = max_match
                self.error_total += self.coins[min_match]

        # Sum total and return closest matched coin
        self.money_total += self.coins[closest_match]
        return closest_match

    def _find_in_dictionary(self, target, haystack):
        """
        Searches in dictionary for value that is closest to target
        """
        return min(haystack, key=lambda search:
            abs(float(haystack[search]) - target))
