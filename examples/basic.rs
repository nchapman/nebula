use std::{io::Write, sync::Arc};

use nebula::{
    options::{ContextOptions, ModelOptions},
    Model,
};

fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!(
            "usage:\n\t{} <model_file_path> \"<promt>\" <n_len = 150>",
            args[0]
        );
        return;
    }
    let model_file_name = args[1].clone();
    let prompt = args[2].clone();
    let mut n_len = None;
    if args.len() > 3 {
        if let Ok(ss) = str::parse(&args[3]) {
            n_len = Some(ss);
        }
    }
    //init nebula
    nebula::init(std::path::PathBuf::from(
        "backends/llama_cpp/llama-cpp-sys/dist",
    ))
    .unwrap();
    println!("model loading...");
    let total_size = 1000;
    let pb = indicatif::ProgressBar::new(total_size);
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}]",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_position(0);
    let pbb = pb.clone();
    let model_options = ModelOptions::builder().n_gpu_layers(10).build();
    let model = Model::new_with_progress_callback(model_file_name, model_options, move |a| {
        pbb.set_position((a * 1000.0) as u64);
        std::thread::sleep(std::time::Duration::from_millis(12));
        true
    })
    .unwrap();
    pb.finish_with_message("done");

    let ctx_options = ContextOptions::default();

    let mut ctx = model.context(ctx_options).unwrap();

    ctx.eval(vec![
        serde_json::json!({
            "role": "user",
            "content": "My family is visiting San Francisco for the first time. Can you create a list of sights we must see and activities we must do?"
        }).try_into().unwrap(),
        serde_json::json!({
            "role": "assistant",
            "content": "Certainly! San Francisco is a vibrant and diverse city with many sights to see and activities to enjoy. Here are some must-see attractions and activities for your family:
Golden Gate Bridge: A world-famous landmark and a stunning example of Art Deco architecture. Walk across the bridge or take a bike ride for a great view of the city.
Fisherman's Wharf: A popular tourist destination with a variety of shops, restaurants, and attractions. Be sure to see the sea lions at Pier 39 and visit the Aquarium of the Bay.
Alcatraz Island: Take a ferry to Alcatraz Island and explore the former prison, which is now a popular tourist attraction.
Cable Car Ride: Take a ride on a historic San Francisco cable car for stunning views of the city and a unique experience.
Chinatown: Explore one of the largest Chinatowns in the United States, with its vibrant culture and a variety of shops, restaurants, and markets.
Lombard Street: Visit the famous Lombard Street, known for its steep, winding roads and beautiful gardens.
San Francisco Museum of Modern Art (SFMOMA): A must-see for art enthusiasts, featuring a diverse collection of modern and contemporary art.
Golden Gate Park: Enjoy a day at this sprawling urban park, with its gardens, museums, and playgrounds.
Exploratorium: A hands-on science museum that encourages learning and exploration through interactive exhibits.
Ferry Building Marketplace: Visit this historic building for a variety of local food vendors, farmers' markets, and artisanal shops.
Union Square: A bustling shopping district with upscale stores, theaters, and dining options.
Ghirardelli Square: Visit the Ghirardelli Chocolate Factory and take a tour to learn about the chocolate-making process, or just enjoy a sweet treat.
Presidio and Golden Gate National Recreation Area: Hike, bike, or simply enjoy the natural beauty of the Presidio and the national park.
Visit a local sports event: Attend a San Francisco Giants baseball game or a San Francisco 49ers football game, depending on the season.
Remember to check the opening hours, availability, and any COVID-19-related restrictions or guidelines before visiting any of these attractions. Enjoy your time in San Francisco!
"
        }).try_into().unwrap(),
        serde_json::json!({
            "role": "user",
            "content": "Great, thanks! Suggest other places"
        }).try_into().unwrap(),
        serde_json::json!({
            "role": "assistant",
            "content": "Certainly! Here are some additional places to explore in San Francisco:
15. Haight-Ashbury: Explore the historic Haight-Ashbury neighborhood, known for its counterculture history and hippie vibe. Visit the Grateful Dead House, Golden Gate Park, and a variety of boutique shops and restaurants.
16. Mission District: This vibrant neighborhood is famous for its colorful murals, street art, and diverse cultural scene. Visit the Mission Dolores Park for stunning views of the city and explore the many street food vendors.
17. Napa Valley: If you have time, consider a day trip to Napa Valley, about an hour north of San Francisco, for wine tasting and scenic views.
18. San Francisco Zoo: A great option for families, the San Francisco Zoo offers a variety of exhibits, including African animals, aquarium, and a children's zoo.
19. Palace of Fine Arts: A stunning architectural landmark with a peaceful lagoon, perfect for a romantic stroll or a picnic.
20. Muir Woods National Monument: Located just north of San Francisco, Muir Woods is home to ancient redwood trees and offers hiking trails and scenic views.
Remember to plan your activities based on your family's interests and time constraints. Enjoy exploring San Francisco!"
        }).try_into().unwrap(),
        serde_json::json!({
            "role": "user",
            "content": "Thank you! Suggest yet another places"
        }).try_into().unwrap(),
        serde_json::json!({
            "role": "assistant",
            "content": " Sure! Here are some more places to explore in San Francisco:
21. Alamo Square: Visit the iconic Painted Ladies Victorian homes and enjoy a panoramic view of the city from the top of the hill.
22. Ferry Building Farmers Market: Visit the farmers market on Saturdays for fresh produce, artisanal foods, and local crafts.
23. Seward's Secret Staircases: Explore the hidden staircases of Seward's Secret Stairs, a community-led project that connects the neighborhood through a network of public art installations.
24. Golden Gate Park Japanese Tea Garden: Relax and enjoy the tranquil atmosphere of this beautiful Japanese garden.
25. San Francisco Museum of the African Diaspora (MOAD): Explore the history, culture, and art of the African Diaspora through interactive exhibits and engaging programs.
26. Fort Mason: Visit Fort Mason, a historic military installation turned park, and enjoy a picnic or a walk along the waterfront.
27. The Wave Organ: A unique and interactive art installation made from reclaimed materials, located on the San Francisco Bay.
28. Civic Center: Visit the grand Civic Center, featuring the City Hall and the main public library, for a taste of the city's architecture and history.
29. North Beach: Explore the city's Little Italy, known for its Italian heritage and a variety of restaurants, cafes, and shops.
30. Golden Gate National Recreation Area Marin Headlands: Enjoy breathtaking views of the Golden Gate Bridge and the Marin County coastline.
Remember to check the opening hours, availability, and any COVID-19-related restrictions or guidelines before visiting any of these attractions. Have a great time exploring San Francisco!"
        }).try_into().unwrap(),
        serde_json::json!({
            "role": "user",
            "content": "Thank you! Now suggest other cities that I may visit after visiting San Francisco"
        }).try_into().unwrap(),
    ]).unwrap();

    let po = nebula::options::PredictOptions::builder()
        .token_callback(Arc::new(Box::new(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
            true
        })))
        .temp(0.8);
    let po = if let Some(ss) = n_len {
        po.max_len(ss).build()
    } else {
        po.build()
    };
    let mut p = ctx.predict(po);
    p.predict().unwrap();
    println!("");
}
