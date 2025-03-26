def run_av_diagnostics():
    import av
    av.open("video_data/brush_hair/Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np1_ba_goo_4.avi")
    print(get_video_backend())
    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, 'pict_type'):
      print("Nah")

run_av_diagnostics()

