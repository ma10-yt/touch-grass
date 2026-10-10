import exifread

def extract_exif_data(file_path: str) -> dict:
    result = {}
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if 'Image DateTime' in tags:
                result['datetime'] = str(tags['Image DateTime'])
            
            # Extract GPS if possible, but keep it simple
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                result['has_gps'] = True
                
    except Exception:
        pass
    return result

