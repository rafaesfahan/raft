import numpy as np
from math import radians, sin, cos, sqrt, atan2
from enum import Enum

# Latitude is specified in degrees within the range [-90, 90]. 
# Longitude is specified in degrees within the range [-180, 180].

def degrees_to_radians(some_degrees_angle):
    radians_angle = some_degrees_angle * np.pi / 180.0
    return radians_angle

def radians_to_degrees(some_radians_angle):
    degrees_angle = some_radians_angle * 180.0 / np.pi
    return degrees_angle
    
def angle_to_complex_encoding(degrees):
    """
    Angle representation (lat/long) fed into models can have sudden sharp discontinuities, to aleviate this we can use a "complex number" 
    representation of each of the coordinate in the coordinate pair. Method transforms either Lattitude or Longgitude into "complex reperesentaion"
    """
    radians = degrees_to_radians(degrees)
    complex_x = np.cos(radians)
    complex_y = np.sin(radians)
    return complex_x, complex_y

def complex_number_to_degrees(complex_x, complex_y):
    radians = np.arctan2(complex_y, complex_x)
    degrees = radians_to_degrees(radians)
    ###degrees = (degrees + 360) % 360.0
    return degrees


# def haversine(lat1, lon1, lat2, lon2):
#     """
#     Calculate the disance in kilometers from (lat1, lon1) to (lat2, lon2)
#     """
#     # Radius of the Earth in kilometers
#     R = 6371.0

#     # Convert latitude and longitude from degrees to radians
#     lat1 = radians(lat1)
#     lon1 = radians(lon1)
#     lat2 = radians(lat2)
#     lon2 = radians(lon2)

#     # Differences in coordinates
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     # Haversine formula
#     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))

#     # Distance in kilometers
#     distance = R * c

#     return distance


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance in kilometers from (lat1, lon1) to (lat2, lon2).
    Supports both scalar and numpy array inputs.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance


class CoordinateEnum(Enum):
    LatLongCoordinates = 1
    ComplexCoordinates = 2
    EmbeddingCoordinates = 3
    PolarCoordinates = 4

    def __str__(self):
        return self.name

    @classmethod
    def convert(cls, item):
        #print(f"Attempting to convert: {item} (type: {type(item)})")
        
        if isinstance(item, cls):
            #print("It's already a CoordinateEnum.")
            return item
        if isinstance(item, str) and item in cls.__members__:
            #print(f"Found '{item}' in __members__.")
            return cls.__members__[item]
        if isinstance(item, int):
            for member in cls:
                if member.value == item:
                    #print(f"Integer {item} matches {member}.")
                    return member
        raise KeyError(f"'{item}' is not a valid {cls.__name__}, item is of type: {type(item)}")

    @classmethod
    def convert_to_str(cls, item):
        return str(cls.convert(item))


def helper_get_coordinate_system_based_on_enum(coordinate_enum):

    # Ensure coordinate_enum is converted to the appropriate enum type
    coordinate_enum = CoordinateEnum.convert(coordinate_enum)
    
    coordinate_system = None
    if coordinate_enum == CoordinateEnum.LatLongCoordinates:
        coordinate_system = ["Latitude", "Longitude"]
    elif coordinate_enum == CoordinateEnum.ComplexCoordinates:
        coordinate_system = ["lat_complex_x", "lat_complex_y", "long_complex_x", "long_complex_y"]
    else:
        raise NotImplementedError(f"Coordinate system {coordinate_enum} Not implemented")
    
    return coordinate_system
