"""
Compatibility module for imghdr, which was removed in Python 3.13.
This provides basic image format detection functionality.
"""

from pathlib import Path


def what(file_path, h=None):
    """
    Identify image type based on file content or signature bytes.
    
    Args:
        file_path: Path to the image file
        h: Optional bytes data to check (if provided, file_path is ignored)
    
    Returns:
        Image format string (e.g., 'jpeg', 'png', 'gif') or None if format is unknown
    """
    
    if h is None:
        try:
            with open(file_path, 'rb') as f:
                h = f.read(32)
        except (OSError, IOError):
            return None
    
    # Check for different image formats by magic numbers
    if h[:2] == b'\xff\xd8' and h[2:3] == b'\xff':
        return 'jpeg'
    elif h[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    elif h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    elif h[:4] == b'RIFF' and h[8:12] == b'WEBP':
        return 'webp'
    elif h[:4] == b'BM':
        return 'bmp'
    elif h[:4] in (b'MM\x00\x2a', b'II\x2a\x00'):
        return 'tiff'
    elif h[:3] == b'ICO':
        return 'ico'
    elif h[:6] == b'ftypisom' or h[:6] == b'ftypmp42':
        return 'heic'
    
    return None
