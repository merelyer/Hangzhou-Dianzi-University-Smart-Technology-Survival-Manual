idf_component_register(SRCS "scurve.c" "compress.c" "table.c" "solve.c" "scramble.c" "optimizer.c" "motion.c" "color_detect.c" "main.c"
                    INCLUDE_DIRS ".")
                    
spiffs_create_partition_image(storage ../data FLASH_IN_PROJECT)
