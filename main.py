import argparse
from counter import coins


def main():
    """
    Runs money counter. Change image file to run here.
    To change calibration image, edit function:
    calibrate_coins() in 'money/counter.py'

    Usage:
    python main.py image.jpg
    """
    parser = argparse.ArgumentParser(description='Counts coins')
    parser.add_argument('image', metavar='image', type=str,
        help='path to image file')
    args = parser.parse_args()

    # Create and calibrate money counter
    coin_counter = coins.CoinCounter()

    # # # Test 1 only coins
    coin_counter.count_coins(args)
    print "Total: $%.2f" % coin_counter.money_total
    print "Error: $%.2f" % coin_counter.error_total

if __name__ == '__main__':
    main()
