use core::fmt;

use tracing::{trace, warn};

use crate::{ppm::PPMFile, vectors::*, Canvas, Color};

pub fn main() {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;

    let mut canvas = Canvas::<WIDTH, HEIGHT>::new();

    let color = Color::white();

    let world = World {
        wind: SpaceElement::<Vector>::new(-0.0001, 0., 0.),
        gravity: SpaceElement::<Vector>::new(0., -0.098, 0.),
    };

    let mut projectile = Projectile {
        position: SpaceElement::<Point>::new(0., 0., 0.),
        velocity: SpaceElement::<Vector>::new(0.01, 0.03, 0.).normalize() * 10.,
    };

    while projectile.position.is_sign_positive() {
        trace!("{projectile}");

        if let Err(error) =
            canvas.write_coordinates(&Coordinates::from(projectile.position), &color)
        {
            warn!("{error}");
        }

        projectile.tick(&world);
    }

    PPMFile::write(
        &canvas.width,
        &canvas.height,
        &canvas.pixels,
        "./output/chapter_2.ppm",
    )
    .unwrap();
}

struct World {
    pub wind: SpaceElement<Vector>,
    pub gravity: SpaceElement<Vector>,
}

struct Projectile {
    pub position: SpaceElement<Point>,
    pub velocity: SpaceElement<Vector>,
}

impl Projectile {
    pub fn tick(&mut self, world: &World) {
        self.position = self.velocity + self.position;
        self.velocity = self.velocity + world.gravity + world.wind;
    }
}

impl fmt::Display for Projectile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Projectile(position: {},  velocity: {})",
            self.position, self.velocity
        )
    }
}
