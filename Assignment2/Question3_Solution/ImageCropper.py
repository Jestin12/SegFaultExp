import cv2
import numpy as np
import os
from pathlib import Path


class ImageCropper:
    def __init__(self, SaveDirectory, InputDirectory):
        self.SaveDirectory = SaveDirectory
        self.InputDirectory = InputDirectory

    def Crop(self, ImageName):

        image_path = self.InputDirectory + ImageName

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: The file at {image_path} does not exist!")
        else:
            print(f"Image path: {image_path}")
            # Load the image
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error loading image at path: {image_path}")
            else:
                print("Image loaded successfully")
                
                image = cv2.resize(image, ( int(image.shape[1]/2), int(image.shape[1]/2) ) )

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                inverted = cv2.bitwise_not(blurred)

                edges = cv2.Canny(inverted, 35, 150)
            
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create a copy of the original image for visualization
                output_image = image.copy()

                # Loop over the contours and crop the traffic signs
                cropped_signs = []
                for contour in contours:
                    # Approximate the contour to a polygon and get bounding box
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    # print("This is the contour value", contour)

                    # Check if the contour is large enough to be a traffic sign
                    if cv2.contourArea(contour) > 2000:  
                        x, y, w, h = cv2.boundingRect(approx)

                        # Crop the detected traffic sign
                        cropped_sign = image[y:y+h, x:x+w]
                        cropped_signs.append(cropped_sign)

                        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                save_directory = self.SaveDirectory

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                ImagePaths = []

                # Assuming you already have cropped_signs from the contour detection code
                for i, cropped_sign in enumerate(cropped_signs):
                    save_path = os.path.join(save_directory, f"cropped_{ImageName}_{i+1}.png")
                    cv2.imwrite(save_path, cropped_sign)
                    print(f"Saved cropped sign {i+1} at {save_path}")
                    ImagePaths.append(f"cropped_{ImageName}_{i+1}.png")

                cv2.imshow("Detected Traffic Signs", output_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return ImagePaths


def main():
    in1 = ImageCropper('/home/jestin/tron5A2/ML/Question3_Solution/cropped_images', '/home/jestin/tron5A2/ML/Question3_Solution/images/')
    in1.Crop("image7.png")

if __name__ == "__main__":
    main()