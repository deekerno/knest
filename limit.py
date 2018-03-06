from PIL import Image
import img_cmp
import os

DIFF_THRES = 20
LIMIT = 2

# this function uses PIL to handles image formats
# returns True if file is an image, otherwise False


def img_handler(img_path):
    try:
        # file is an image
        img = Image.open(img_path)
        img.close()
        return True

    except IOError:
        # file is not an image
        return False


# runs through an entire folder of images and limits the number of similar images
# limited to 3 per similar images
# difference threshold for imagehash is 18
# dir_path is the path of the images given
# des_path is the path where all the accepted images will be saved
def limit(dir_path, des_path):
    # initialized variables
    std = ''
    count = 0

    # go through directory of images
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)

        # file is an image
        if img_handler(img_path):
            # if there is no comparison standard, set one
            # will make this into a separate function later, cause I'm lazy
            if std == '':
                # set standard info
                std = img_path
                std_name = filename
                std_hash = img_cmp.calc_hash(std)
                count = 0

                # save the standard to the directory
                im = Image.open(std)
                im.save(os.path.join(des_path, filename))

                print("Current standard for comparison: ", filename)

            else:
                # calcuate hash for comparing image
                cmp_hash = img_cmp.calc_hash(img_path)
                # compare to standard image
                diff = img_cmp.compare(std_hash, cmp_hash)
                # image is similar to standard
                if diff <= DIFF_THRES:
                        # add to destination folder if there are <= 3 similar
                        # images already
                    if count < LIMIT:
                        print(filename, " is similar to ", std_name,
                              " with ", diff, " difference and ACCEPTED")
                        # save the similar file to the directory
                        im = Image.open(img_path)
                        im.save(os.path.join(des_path, filename))

                    # else, ignore image and move on until we come upon a
                    # non-similar image
                    else:
                        print(filename, " is similar to ", std_name,
                              " with ", diff, " difference but REJECTED")

                    # keep track of number similar images
                    count += 1

                # a non-similar image has been found, make it the new standard
                # for comparison
                else:
                        # update new standard info
                    std = img_path
                    std_name = filename
                    std_hash = img_cmp.calc_hash(std)
                    count = 0

                    # save the standard to the directory
                    im = Image.open(std)
                    im.save(os.path.join(des_path, filename))

                    print("Found a nonsimilar image with difference ", diff)
                    print("Current standard for comparison: ", filename)


if __name__ == '__main__':
	limit('/users/ayylmao/downloads/test2/', 'results/')
