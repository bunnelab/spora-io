import zarr

ZARR_COMPRESSOR = zarr.codecs.BloscCodec(cname='lz4', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
MAX_BLOSC_BUFFER_BYTES = 2_147_483_647
