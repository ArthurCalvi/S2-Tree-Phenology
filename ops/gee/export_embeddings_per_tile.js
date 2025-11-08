// Google Earth Engine script: Export 2023 Satellite Embeddings per training tile
// Instructions:
// 1) Upload your tiles as a FeatureCollection asset (EPSG:2154). Use the
//    tiles from results/datasets/tiles_2_5_km_final.parquet converted to
//    GeoJSON or SHP and ingested into an asset, ensuring a tile_id property
//    exists (integer, zero-based or sequential).
// 2) Set ASSET_TILES below to your asset path.
// 3) Choose an index range and run exportTile(i) for each index. Repeat/batch
//    as needed (GEE UI requires manually starting tasks).

// CONFIG
var ASSET_TILES = 'users/USERNAME/tiles_2_5_km_final';  // <-- change this
var YEAR = 2023;
var CRS = 'EPSG:2154';
var SCALE = 10; // meters

// Load tiles and sort by tile_id (required for stable indexing)
var tiles = ee.FeatureCollection(ASSET_TILES)
  .filter(ee.Filter.notNull(['tile_id']))
  .sort('tile_id');

// Load the 2023 embedding image (64-D variant: embedding_0..embedding_63)
var embeddings = ee.ImageCollection('GOOGLE/SATELLITE/EMBEDDING/V1/ANNUAL')
  .filterDate(ee.Date.fromYMD(YEAR,1,1), ee.Date.fromYMD(YEAR,12,31))
  .first()
  .select(['embedding_.*']); // keep GEE default names

// Helper: export one tile by list index
function exportTileByIndex(idx) {
  idx = ee.Number(idx);
  var ft = ee.Feature(ee.List(tiles.toList(100000)).get(idx));
  var tileId = ee.Number(ft.get('tile_id'));
  var region = ft.geometry();

  var img = embeddings
    .reproject({crs: CRS, scale: SCALE})
    .clip(region);

  var filePrefix = ee.String('emb_tile_')
      .cat(tileId.format('%03d'))
      .cat('_').cat(ee.String(YEAR.toString()));

  Export.image.toDrive({
    image: img,
    description: filePrefix.getInfo(),
    fileNamePrefix: filePrefix.getInfo(),
    region: region,
    crs: CRS,
    scale: SCALE,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}

// Usage examples in the Code Editor console:
// exportTileByIndex(0);   // exports first tile
// exportTileByIndex(1);   // exports second tile
// ...
// For batches, call exportTileByIndex for a range (manually or via small loops)
// Note: The GEE UI requires you to start tasks manually.

