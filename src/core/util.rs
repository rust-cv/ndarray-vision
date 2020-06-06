/// Get the centre of a kernel. Determines the pixel to be
/// modified as a window is moved over an image
pub fn kernel_centre(rows: usize, cols: usize) -> (usize, usize) {
    let row_offset = rows / 2 - ((rows % 2 == 0) as usize);
    let col_offset = cols / 2 - ((cols % 2 == 0) as usize);
    (row_offset, col_offset)
}
