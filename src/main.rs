use bevy::prelude::*;

mod util;
mod camera;
mod bigmat;

fn main() {
	App::new()
	.insert_resource(Msaa { samples: 4 })
	.add_plugins(DefaultPlugins)
	.run();
}
