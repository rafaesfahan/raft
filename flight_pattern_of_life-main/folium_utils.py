from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import time
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import folium

import datetime


def create_folium_map(lat1, long1, lat2, long2, border_lat_prop=0.15, border_long_prop=0.15, tiles=None):
    """
    Create folium map object given min/max coordinate points of desired flightpath

    tiles, string: changes colorscheme of how map is rendered ('CartoDB Positron', 'CartoDB Voyager', 'Cartodb dark_matter')

    params:
        lat1: Type, float: latitude_1
        long1: Type, float: longitude_1
        lat2: Type, float: latitude_2
        long2: Type, float: longitude_2
        border_lat_prop: Type, float: Latitude percentage margin extra around rectangle defined by (lat1, long1) and (lat2, long2)
        border_long_prop: Type, float: Longitude percentage margin extra around rectangle defined by (lat1, long1) and (lat2, long2)
        tiles: Type, string: the 'style' of the folim map, default is 'CartoDB Positron'
    """

    if tiles is None:
        #tiles="Cartodb dark_matter"
        tiles = 'CartoDB Positron'
    
    lat1, lat2 = min(lat1, lat2), max(lat1, lat2)
    long1, long2 = min(long1, long2), max(long1, long2)
    
    # Create a map object
    m = folium.Map(location=[(lat1 + lat2) / 2, (long1 + long2) / 2], zoom_start=4, tiles=tiles)
    
    dist_lat = max(lat1, lat2) - min(lat1, lat2)
    dist_long = max(long1, long2) - min(long1, long2)

    # if dist_lat < 0.5:
    #     dist_lat = 1.0
    # if dist_long < 0.5:
    #     dist_long = 1.0
        
    bound_lat1 = lat1 - border_lat_prop * dist_lat
    bound_lat2 = lat2 + border_lat_prop * dist_lat
    bound_long1 = long1 - border_long_prop * dist_long
    bound_long2 = long2 + border_long_prop * dist_long
    
    
    # Fit the map to the specified bounds
    m.fit_bounds([[bound_lat1, bound_long1], [bound_lat2, bound_long2]])

    return m


# folium.PolyLine(locations=example_coordinates, color='blue', weight=1.4, opacity=1).add_to(m)

# Save the map as an HTML file
#m.save('/Users/aleksandranikevich/Desktop/AircraftTrajectory/maps/map.html')


    
def get_map_image(m, file_base = None, save_map_name = None, firefox_dir = None, firefox_binary = None):
    """
    Given folium map object, first save the html in a directory then crop an image and save the png
    """

    # convinence 
    if firefox_dir is None:
        firefox_dir = '/Users/aleksandranikevich/Desktop/AircraftTrajectory/geckodriver'
    if firefox_binary is None:
        firefox_binary = '/Applications/Firefox.app/Contents/MacOS/firefox'  # Update this path to your Firefox binary location

    # Selenium driver setup 
    os.environ['MOZ_HEADLESS'] = '1' 
    options = Options()
    options.headless = True
    options.binary_location = firefox_binary
    service = Service(firefox_dir)
    driver = webdriver.Firefox(service=service, options=options)

    
    # Convert the file path to a URL
    if file_base is None:
        file_base  = '/Users/aleksandranikevich/Desktop/AircraftTrajectory/maps/'
    if save_map_name is None:
        first_date = datetime.datetime(1970, 1, 1)
        time_since = datetime.datetime.now() - first_date
        seconds = int(time_since.total_seconds())
        save_map_name = str(seconds)
    file_path = file_base + f'map_{save_map_name}.html'
    file_url = 'file://' + os.path.abspath(file_path)

    html_map_path = file_base + f'map_{save_map_name}.html'
    png_map_path = file_base + f'map_{save_map_name}.png'
    m.save(html_map_path) #('/Users/aleksandranikevich/Desktop/AircraftTrajectory/maps/map.html')
    
    driver.get(file_url)
    time.sleep(3)
    
    # Save the screenshot
    driver.save_screenshot(png_map_path)
    driver.quit()

    return html_map_path, png_map_path



# def png_path_to_fig(image_path, main_title):
#     # Read the image
#     img = mpimg.imread(image_path)
    
#     # Get the dimensions of the image
#     height, width, _ = img.shape
    
#     # Convert dimensions from pixels to inches (assuming 100 DPI)
#     dpi = 100
#     fig_width = width / dpi
#     fig_height = height / dpi
    
#     # Create a figure with the same size as the image
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
#     # Display the image
#     ax.imshow(img)
    
#     # Remove axes for a cleaner look
#     ax.axis('off')

#     # Add a main title if provided
#     if main_title is not None:
#         fig.suptitle(main_title)
    
#     return fig

def png_path_to_fig(image_path, main_title=None, num_rows=None, num_cols=None):
    # Read the image
    img = mpimg.imread(image_path)
    
    # Get the dimensions of the image
    height, width, _ = img.shape
    
    # Convert dimensions from pixels to inches (assuming 100 DPI)
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    
    # Check if num_rows and num_cols are specified
    if num_rows is not None and num_cols is not None:
        # Create a figure with the specified number of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)
        
        # Flatten the axs array if it's 2D for easier indexing
        if num_rows > 1 or num_cols > 1:
            axs = axs.flatten()
        
        # Display the image in the first subplot
        axs[0].imshow(img)
        axs[0].axis('off')
        
        # Add a main title if provided
        if main_title is not None:
            fig.suptitle(main_title)
        
        return fig, axs
    else:
        # Create a figure with the same size as the image
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display the image
        ax.imshow(img)
        ax.axis('off')
        
        # Add a main title if provided
        if main_title is not None:
            fig.suptitle(main_title)
        
        return fig, ax