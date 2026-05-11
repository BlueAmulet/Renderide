//! World text unlit (Unity shader asset `TextUnit`): MSDF / SDF / raster font atlas in world space.
//!
//! Compatibility route that shares the `textunlit` shader body. The host resolves
//! `TextUnit` to `textunit_default` / `textunit_multiview`; the WGSL source matches
//! `textunlit.wgsl` verbatim via the `//#source_alias` directive below.
//#source_alias textunlit
