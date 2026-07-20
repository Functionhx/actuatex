# Livox Mid-360 sensor asset

This directory deliberately separates three things that are often conflated
in lidar demos:

1. `usd/mid360_visual.usda` is a visual-only, open teaching shell. Its optical
   origin follows Livox's mechanical drawing: 47 mm above the bottom face.
2. RTX ray generation is implemented in
   `backends/isaac_lab/tinymal_lab/mid360_rtx.py` from Livox's full official
   800,000-point scan table. It is not a spinning 2D approximation.
3. ROS message packing is implemented in `actuatex_navigation`. Standard
   output uses the same packed `PointXYZRTLT` fields as `livox_ros_driver2`;
   the official `CustomMsg` is also emitted when that package is installed.

The visual shell approximates the public 65 × 65 × 60 mm envelope and is not
a metrology-grade reconstruction. The official STEP is intentionally not
redistributed because its download page does not grant a repository-friendly
redistribution licence. Fetch it locally with:

```bash
HTTPS_PROXY=http://127.0.0.1:7890 scripts/fetch_mid360_cad.sh
```

The downloaded CAD remains under ignored `artifacts/vendor_assets/`. See
`source.json` for provenance, hashes, specifications, and the exact distinction
between official data and our procedural approximation.
