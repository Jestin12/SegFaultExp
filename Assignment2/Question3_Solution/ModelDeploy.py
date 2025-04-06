import json
from datetime import datetime
from cPedestrian import Pedestrian
import os

'''
*************************** ModelDeploy.py ***************************************

Filename:       cModelDeploy.py
Author:         Neel, Jestin

Description:    This script instantiates the Pedestrian class and passes in various
                images into the Pedestrian class to classify, uses the Pedestrian methods
                to produce a .csv with the classification record of all the images passed 
                in as well as prints to the terminal the classification results.

Dependencies:   cPredestrian    datetime    json    os

************************************************************************************
'''

def process_all_rosbags(ModelPath, OutputDirectory, InputDirectory, display_first_image=True):
    
    '''
    Instantiates the Pedestrian class, iterates through all "ros_bag" folders found in the InputDirectory for input images 
    which it then uses the Pedestrian class' methods to find the signs and red cylinders within the input image and records
    the data in a .csv file.

    Also, displays the first image it processes with rectangular borders and labels identifying the red cylinders and signs
    the ML model finds in the image.

    Inputs:
        ModelPath (string): File path to the machine learning model .pth file, file path should relative to the Question3_Solutions
                            directory
        
        OutputDirectory (string):   File path to the directory where the cropped images produced from the input images should be stored, 
                                    file path should relative to the Question3_Solutions directory

        InputDirectory (string):    File path to the directory where the input images are, input images should be stored in folders
                                    with "rosbag" in their name, file path should relative to the Question3_Solutions
                                    directory
                                
        display_first_image (bool): True/False to determine if an image should be shown as a sample with its border rectangles and
                                    labels around the red cylinders and signs
    '''

    detector = Pedestrian(ModelPath, OutputDirectory, InputDirectory)
    
    # Get all rosbag directories in the InputDirectory, change the if-statement to generalise the function
    rosbag_dirs = [d for d in os.listdir(InputDirectory) if os.path.isdir(os.path.join(InputDirectory, d)) and "rosbag" in d]
    
    total_rosbags = len(rosbag_dirs)
    print(f"Found {total_rosbags} rosbag directories to process")
    
    first_image_key = None
    
    # Process each rosbag directory
    for i, rosbag_name in enumerate(rosbag_dirs):
        print(f"\nProcessing rosbag {i+1}/{total_rosbags}: {rosbag_name}")
        rosbag_path = os.path.join(InputDirectory, rosbag_name)
        
        # Get all image files in the rosbag directory
        image_files = [f for f in os.listdir(rosbag_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images in {rosbag_name}")
        
        # Process each image
        for j, image_name in enumerate(image_files):
            print(f"  Processing image {j+1}/{len(image_files)}: {image_name}")
            detector.Classify(image_name, rosbag_name)
            
            # Store the first image key for display later
            if first_image_key is None:
                first_image_key = f"{rosbag_name}/{image_name}"
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    detector.save_detection_stats(f"Question3_Solution/detection_stats_{timestamp}.csv")
    
    # Display the first processed image with bounding boxes and labels
    if display_first_image and first_image_key is not None:
        print(f"\nDisplaying results for the first processed image: {first_image_key}")
        detector.display_processed_image(first_image_key)
    
    return detector.detection_stats


def main():
    InputDirectory = 'Question3_Solution/images/'  # Directory containing rosbag folders
    OutputDirectory = 'Question3_Solution/cropped_images/'  # Output directory for cropped signs
    ModelPath = "Question3_Solution/Models/best_model_final.pth"    # File path to the pytorch machine learning model
    
    # Process all rosbags and get statistics, then display the first image
    stats = process_all_rosbags(ModelPath, OutputDirectory, InputDirectory, display_first_image=True)
    print("\nProcessing completed!")

    output = input("Should I print the results to the terminal? (Y) ")
    if output == "Y":
        TerminalOutput = json.dumps(stats, indent = 4)

        print(TerminalOutput)
    
    print("Code executed")

if __name__ == '__main__':
    main()