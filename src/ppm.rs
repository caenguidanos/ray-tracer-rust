use std::{io::Write, path::Path};

use tracing::info;

use crate::Color;

pub struct PPMFile;

impl PPMFile {
    pub fn write<P>(
        width: &usize,
        height: &usize,
        pixels: &[Color],
        path: P,
    ) -> Result<(), std::io::Error>
    where
        P: Into<String> + AsRef<Path>,
    {
        let mut buffer = std::fs::File::create(&path)?;

        let header = format!("P3\n{width} {height}\n255");
        buffer.write(header.as_bytes())?;
        drop(header);

        for chunk in pixels.chunks(*width) {
            buffer.write(b"\n")?;

            let mut section_chunks = vec![];

            for pixel in chunk.iter() {
                section_chunks.push(format!(
                    "{} {} {}",
                    pixel.r_as_byte(),
                    pixel.g_as_byte(),
                    pixel.b_as_byte()
                ));
            }

            buffer.write(section_chunks.join(" ").as_bytes())?;
        }

        buffer.write(b"\n")?;
        buffer.flush()?;

        info!("Image saved on {}", path.into());

        Ok(())
    }
}
