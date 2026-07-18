
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <math.h>
#include "esp_mac.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "sys/param.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/gpio.h"
#include "spi_flash_mmap.h"
#include "esp_http_server.h"
#include "esp_spiffs.h"
#include "driver/ledc.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_jpeg_common.h"
#include "esp_jpeg_dec.h"
#include "esp_heap_caps.h"
#include "driver/usb_serial_jtag.h"


#include "color_detect.h"
#include "motion.h"
#include "pin_def.h"
#include "solve.h"
#include "optimizer.h"
#include "scramble.h"

typedef enum {
    CMD_DISABLE_ALL,
    CMD_ENABLE_ALL,
    CMD_QUERY_ANGLES,
    CMD_CLAMP_CUBE,
    CMD_RELEASE_CUBE,
    CMD_GO_HOME,
    CMD_MOTIONS,
    CMD_FLIP_TO_FACE_2,   // 翻转到第2个面
    CMD_FLIP_TO_FACE_3    // 翻转到第3个面
} motion_cmd_t;

typedef struct{
    size_t len;
    uint8_t *buf;
} jpeg_buf;

#define SCRAMBLE_SEQUENCE_LEN 128
#define MAIN_CFG_PATH "/spiffs/main.cfg"
#define MOTION_CFG_PATH "/spiffs/motion.cfg"
#define CAMERA_FLIP_SEQUANCE_1 "L1 R-F L0 R2 R+N L*F R0"
#define CAMERA_FLIP_SEQUANCE_2 "R1 L-F R0 L2 L+N R*F L0"
typedef struct{
    int background_discard_frames;  // 上电后需要舍弃的帧
    int stabilization_frames;       // 需要的稳定帧数
    float stabilization_percent;    // 图像静止的像素百分比
    int thresh_diff;                // 像素变化阈值
    int sat_hue_threshold;          // 色块变化阈值
    int change_threshold;           // 变化色块数量阈值（放置魔方）
    int change_threshold_remove;    // 变化色块数量阈值（移除魔方）
    int led_brightness;             // 补光灯亮度
    int camera_ae_level;            // 自动曝光级别 (-5 ~ +5)
    int auto_cube_detect;           // =1开启魔方放置检测 =2开启自动还原
    int performance_test;           // =1上电后测试还原算法的性能
    Point color_detect_points[6];   // 魔方图像边界的坐标
    Point color_detect_quads[18][4];// 每个色块的边界坐标（不存储，在需要时计算）
    char scramble_sequence[SCRAMBLE_SEQUENCE_LEN]; // 固定打乱序列
    int hs_equal_level;             // 判断魔方色块采集是否稳定的依据
    char sta_ssid[32];           // STA模式SSID
    char sta_password[64];       // STA模式密码
}main_cfg;

typedef enum {
    CUBE_DETECT_MANUAL = 0,
    CUBE_DETECT_ONLY_CAPTURE_FACE = 1,
    CUBE_DETECT_AUTO = 2,
}auto_cube_detect_mode;

typedef enum{
    STATE_INIT,              // 初始状态，舍弃前10帧
    STATE_BACKGROUND,        // 采集背景颜色
    STATE_DETECTING,         // 检测魔方放置
    STATE_STABILIZING,       // 检测稳定性
    STATE_CAPTURE_FACE_1,    // 捕获第1个面
    STATE_FLIP_TO_FACE_2,    // 翻转到第2个面
    STATE_CAPTURE_FACE_2,    // 捕获第2个面
    STATE_FLIP_TO_FACE_3,    // 翻转到第3个面
    STATE_CAPTURE_FACE_3,    // 捕获第3个面
    STATE_SOLVE,             // 正在解魔方
    STATE_DETECTING_REMOVE   // 检测魔方移除
}camera_stat_t;

static QueueHandle_t motion_cmd_queue = NULL;// 运动控制队列句柄
static QueueHandle_t jpeg_queue = NULL; // 预览图像队列

static const char *TAG = "main";

static main_cfg cfg;
static httpd_handle_t camera_httpd = NULL;  // 主服务器
static httpd_handle_t stream_httpd = NULL;  // 流服务器
static uint8_t g_color_data[54][3] = {0};   // [H, S, V] 存储颜色数据
static atomic_bool g_flip_completed = ATOMIC_VAR_INIT(false); // 完成魔方翻转的标志位
static atomic_bool g_motor_enable = ATOMIC_VAR_INIT(false);// 点击开启标志位
static atomic_bool g_motor_error = ATOMIC_VAR_INIT(false);// 电机错误标志位
static atomic_bool g_restore = ATOMIC_VAR_INIT(false); // 手动触发魔方还原
static atomic_int g_camera_state = ATOMIC_VAR_INIT(0);
static atomic_int g_time_cost_us = ATOMIC_VAR_INIT(0);
static char g_motion_sequence[MOTION_SEQUENCE_LEN] = {0}; // 运动控制字符串，1KB
typedef struct{
    int camera_flip; // 翻面耗时
    int motion_sequence;// 还原耗时
    int other;// 其他耗时（计算+摄像头采集图像）
} estimate_time;
static estimate_time g_estimate_time={400, 5000, 300};

// 估算图像采集翻面时间
static int update_camera_flip_time(void){
    int time = get_time_motions_str(CAMERA_FLIP_SEQUANCE_1);
    if (time < 0 || time > 32768){
        ESP_LOGE(TAG, "计算执行时间失败：%s", CAMERA_FLIP_SEQUANCE_1);
        return -1;
    }
    g_estimate_time.camera_flip = (int16_t)time;
    return 0;
}

static char g_cube_str[55];

// PWM配置
#define LEDC_TIMER          LEDC_TIMER_0
#define LEDC_MODE           LEDC_LOW_SPEED_MODE
#define LEDC_CHANNEL        LEDC_CHANNEL_0
#define LEDC_DUTY_RES       LEDC_TIMER_10_BIT  // 10位分辨率，占空比范围0-1023
#define LEDC_FREQUENCY      40000               // PWM频率40kHz

#define EXAMPLE_ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define EXAMPLE_ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD
#define EXAMPLE_ESP_WIFI_CHANNEL   CONFIG_ESP_WIFI_CHANNEL
#define EXAMPLE_MAX_STA_CONN       CONFIG_ESP_MAX_STA_CONN

static inline int min_int(int a, int b) { return (a < b) ? a : b; }
static inline int max_int(int a, int b) { return (a > b) ? a : b; }

// 从指定路径加载配置文件
static int load_main_cfg(const char* file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        ESP_LOGE(TAG, "无法打开配置文件: %s", file_path);
        return -1;
    }

    // 定义所有配置项的标记
    typedef struct {
        bool background_discard_frames;
        bool stabilization_frames;
        bool stabilization_percent;
        bool thresh_diff;
        bool sat_hue_threshold;
        bool change_threshold;
        bool change_threshold_remove;
        bool led_brightness;
        bool camera_ae_level;
        bool color_detect_points;
        bool auto_cube_detect;
        bool performance_test;
        bool scramble_sequence;
        bool hs_equal_level;
        bool sta_ssid;
        bool sta_password;
    } ConfigFlags;
    
    ConfigFlags flags = {0};
    char line[256];
    int line_num = 0;
    int error_count = 0;

    while (fgets(line, sizeof(line), file)) {
        line_num++;
        // 移除行尾的换行符（兼容LF、CR、CRLF）
        char *newline = strpbrk(line, "\r\n");
        if (newline) *newline = '\0';

        // 跳过注释行和空行
        if (line[0] == '#' || line[0] == '\0') continue;
        
        char key[64], value[256];
        if (sscanf(line, "%63[^=]= %255[^\n]", key, value) == 2) {
            // 去除key中的尾部空格
            char *end = key + strlen(key) - 1;
            while (end > key && isspace((int)*end)) end--;
            *(end + 1) = '\0';
            // 解析并标记找到的配置项
            if (strcmp(key, "BACKGROUND_DISCARD_FRAMES") == 0) {
                cfg.background_discard_frames = atoi(value);
                flags.background_discard_frames = true;
            }
            else if (strcmp(key, "STABILIZATION_FRAMES") == 0) {
                cfg.stabilization_frames = atoi(value);
                flags.stabilization_frames = true;
            }
            else if (strcmp(key, "STABILIZATION_PERCENT") == 0) {
                cfg.stabilization_percent = atof(value);
                flags.stabilization_percent = true;
            }
            else if (strcmp(key, "THRESH_DIFF") == 0) {
                cfg.thresh_diff = atoi(value);
                flags.thresh_diff = true;
            }
            else if (strcmp(key, "SAT_HUE_THRESHOLD") == 0) {
                cfg.sat_hue_threshold = atoi(value);
                flags.sat_hue_threshold = true;
            }
            else if (strcmp(key, "CHANGE_THRESHOLD") == 0) {
                cfg.change_threshold = atoi(value);
                flags.change_threshold = true;
            }
            else if (strcmp(key, "CHANGE_THRESHOLD_REMOVE") == 0) {
                cfg.change_threshold_remove = atoi(value);
                flags.change_threshold_remove = true;
            }
            else if (strcmp(key, "LED_BRIGHTNESS") == 0) {
                cfg.led_brightness = atoi(value);
                flags.led_brightness = true;
            }
            else if (strcmp(key, "CAMERA_AE_LEVEL") == 0) {
                cfg.camera_ae_level = atoi(value);
                flags.camera_ae_level = true;
            }
            else if (strcmp(key, "AUTO_CUBE_DETECT") == 0) {
                cfg.auto_cube_detect = atoi(value);
                flags.auto_cube_detect = true;
            }
            else if (strcmp(key, "PERFORMANCE_TEST") == 0) {
                cfg.performance_test = atoi(value);
                flags.performance_test = true;
            }
            else if (strcmp(key, "HS_EQUAL_LEVEL") == 0) {
                cfg.hs_equal_level = atoi(value);
                flags.hs_equal_level = true;
            }
            else if (strcmp(key, "STA_SSID") == 0) {
                strncpy(cfg.sta_ssid, value, sizeof(cfg.sta_ssid) - 1);
                cfg.sta_ssid[sizeof(cfg.sta_ssid) - 1] = '\0';
                flags.sta_ssid = true;
                ESP_LOGI(TAG, "Parsed STA_SSID: %s", cfg.sta_ssid);
            }
            else if (strcmp(key, "STA_PASSWORD") == 0) {
                strncpy(cfg.sta_password, value, sizeof(cfg.sta_password) - 1);
                cfg.sta_password[sizeof(cfg.sta_password) - 1] = '\0';
                flags.sta_password = true;
                ESP_LOGI(TAG, "Parsed STA_PASSWORD: %s", cfg.sta_password);
            }
            else if (strcmp(key, "COLOR_DETECT_POINTS") == 0) {
                char *ptr = value;
                int points_parsed = 0;
                
                for (int i = 0; i < 6; i++) {
                    int n = 0;
                    if (sscanf(ptr, " {%hd, %hd}%n", 
                            &cfg.color_detect_points[i].x, 
                            &cfg.color_detect_points[i].y, &n) != 2) {
                        error_count++;
                        ESP_LOGE(TAG, "Failed to parse point %d", i);
                        break;
                    }
                    points_parsed++;
                    ptr += n;  // 移动到已解析部分之后
                    
                    // 跳过后续逗号和空格（如果有）
                    while (*ptr == ',' || isspace((int)*ptr)) ptr++;
                }
                
                if (points_parsed != 6) {
                    ESP_LOGE(TAG, "仅解析了%d个点，需要6个点", points_parsed);
                    error_count++;
                } else {
                    flags.color_detect_points = true;
                }
            }
            else if (strcmp(key, "SCRAMBLE_SEQUENCE") == 0) {
                // 打乱序列加载
                strncpy(cfg.scramble_sequence, value, SCRAMBLE_SEQUENCE_LEN - 1);
                cfg.scramble_sequence[SCRAMBLE_SEQUENCE_LEN - 1] = '\0';
                flags.scramble_sequence = true;
            }
            else {
                ESP_LOGW(TAG, "未知配置项 '%s' (行 %d)", key, line_num);
            }
        }
        // 忽略其他的情况，例如完全由空格组成的行
        // else {
        //     ESP_LOGW(TAG, "无效配置行: %s (行 %d)", line, line_num);
        //     error_count++;
        // }
    }
    fclose(file);
    // 检查所有必需的配置项
    #define CHECK_FLAG(flag, name) \
        if (!flag) { \
            ESP_LOGE(TAG, "缺少必需的配置项: %s", name); \
            error_count++; \
        }
    #define CHECK_FLAG_WARN(flag, name) \
    if (!flag) { \
        ESP_LOGW(TAG, "缺少非必要的配置项: %s", name); \
    }
    CHECK_FLAG(flags.background_discard_frames, "BACKGROUND_DISCARD_FRAMES");
    CHECK_FLAG(flags.stabilization_frames,      "STABILIZATION_FRAMES");
    CHECK_FLAG(flags.stabilization_percent,     "STABILIZATION_PERCENT");
    CHECK_FLAG(flags.thresh_diff,               "THRESH_DIFF");
    CHECK_FLAG(flags.sat_hue_threshold,         "SAT_HUE_THRESHOLD");
    CHECK_FLAG(flags.change_threshold,          "CHANGE_THRESHOLD");
    CHECK_FLAG(flags.change_threshold_remove,   "CHANGE_THRESHOLD_REMOVE");
    CHECK_FLAG(flags.led_brightness,            "LED_BRIGHTNESS");
    CHECK_FLAG(flags.camera_ae_level,           "CAMERA_AE_LEVEL");
    CHECK_FLAG(flags.color_detect_points,       "COLOR_DETECT_POINTS");
    CHECK_FLAG(flags.auto_cube_detect,          "AUTO_CUBE_DETECT");
    CHECK_FLAG(flags.performance_test,          "PERFORMANCE_TEST");
    CHECK_FLAG(flags.scramble_sequence,         "SCRAMBLE_SEQUENCE");
    CHECK_FLAG(flags.hs_equal_level,            "HS_EQUAL_LEVEL");
    CHECK_FLAG_WARN(flags.sta_ssid,             "STA_SSID");
    CHECK_FLAG_WARN(flags.sta_password,         "STA_PASSWORD");

    #undef CHECK_FLAG
    
    if (error_count > 0) {
        ESP_LOGE(TAG, "配置文件 %s 有 %d 个错误", file_path, error_count);
        return -1;
    }
    return 0;  // 成功
}

// 将配置保存到指定路径（适用于SPIFFS文件系统）
// 将配置保存到指定路径（适用于SPIFFS文件系统）
static int save_main_cfg(const char* file_path) {
    // 1. 只读模式打开文件查找标记位置并计算尾部长度
    FILE *fp = fopen(file_path, "r");
    if (!fp) {
        ESP_LOGE(TAG, "Failed to open file: %s", file_path);
        return -1;
    }
    
    // 查找标记行位置并计算尾部长度
    long marker_pos = -1;
    long tail_length = 0;
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "DO NOT EDIT BELOW THIS LINE")) {
            marker_pos = ftell(fp);
            // 计算尾部长度（从标记到文件结束）
            fseek(fp, 0, SEEK_END);
            tail_length = ftell(fp) - marker_pos;
            break;
        }
    }
    
    if (marker_pos == -1) {
        ESP_LOGE(TAG, "Marker not found");
        fclose(fp);
        return -1;
    }
    fclose(fp);
    
    // 2. 读写模式打开文件
    fp = fopen(file_path, "r+");
    if (!fp) {
        ESP_LOGE(TAG, "Failed to open file for update: %s", file_path);
        return -1;
    }
    
    // 3. 定位到标记后的位置
    if (fseek(fp, marker_pos, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    
    // 4. 使用fprintf直接写入新配置
    long start_pos = ftell(fp); // 记录写入起始位置
    
    // 写入LED亮度
    if (fprintf(fp, "# 补光灯亮度\nLED_BRIGHTNESS = %d\n", cfg.led_brightness) < 0) {
        fclose(fp);
        return -1;
    }
    
    // 写入打乱公式
    if (fprintf(fp, "# 打乱公式\nSCRAMBLE_SEQUENCE = %s\n", cfg.scramble_sequence) < 0) {
        fclose(fp);
        return -1;
    }
    
    // 写入魔方位置点
    if (fprintf(fp, "# 魔方位置\nCOLOR_DETECT_POINTS = ") < 0) {
        fclose(fp);
        return -1;
    }
    
    for (int i = 0; i < 6; i++) {
        if (i > 0) {
            if (fprintf(fp, ", ") < 0) {
                fclose(fp);
                return -1;
            }
        }
        if (fprintf(fp, "{%d, %d}", 
                   cfg.color_detect_points[i].x, 
                   cfg.color_detect_points[i].y) < 0) {
            fclose(fp);
            return -1;
        }
    }
    
    if (fprintf(fp, "\n") < 0) {
        fclose(fp);
        return -1;
    }
    
    // 5. 计算实际写入长度
    long end_pos = ftell(fp);
    int written_length = end_pos - start_pos;
    
    // 6. 计算需要填充的空格数
    int padding_needed = tail_length - written_length;
    
    // 7. 用空格填充多余空间
    if (padding_needed > 0) {
        char padding[128];
        memset(padding, ' ', sizeof(padding));
        
        while (padding_needed > 0) {
            int chunk_size = padding_needed > (int)sizeof(padding) ? sizeof(padding) : padding_needed;
            if (fwrite(padding, 1, chunk_size, fp) != (size_t)chunk_size) {
                ESP_LOGE(TAG, "Padding write failed");
                break;
            }
            padding_needed -= chunk_size;
        }
        
        // 确保最后有换行符
        fputc('\n', fp);
    }
    
    // 8. 确保所有内容已写入
    fflush(fp);
    fclose(fp);
    
    ESP_LOGI(TAG, "Config updated: %s", file_path);
    return 0;
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                                    int32_t event_id, void* event_data)
{
    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t* event = (wifi_event_ap_staconnected_t*) event_data;
        ESP_LOGI(TAG, "station "MACSTR" join, AID=%d",
                 MAC2STR(event->mac), event->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t* event = (wifi_event_ap_stadisconnected_t*) event_data;
        ESP_LOGI(TAG, "station "MACSTR" leave, AID=%d, reason=%d",
                 MAC2STR(event->mac), event->aid, event->reason);
    } else if (event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "STA started");
        esp_wifi_connect();
    } else if (event_id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG, "STA connected to AP");
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
        static int retry_count = 0;
        const int max_retries = 3;
        
        ESP_LOGI(TAG, "STA disconnected from AP, retry %d/%d", retry_count + 1, max_retries);
        
        if (retry_count < max_retries) {
            // 尝试重连
            esp_wifi_connect();
            retry_count++;
        } else {
            // 超过最大重试次数，禁用STA模式
            ESP_LOGW(TAG, "Maximum retry attempts reached, disabling STA mode");
            esp_wifi_set_mode(WIFI_MODE_AP);
            retry_count = 0;
        }
    }
}
static void ip_event_handler(void* arg, esp_event_base_t event_base,
                                 int32_t event_id, void* event_data)
{
    if (event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

static void wifi_init_sta_ap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    // 创建STA和AP的netif接口
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg_init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg_init));

    // 注册事件处理器
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &ip_event_handler,
                                                        NULL,
                                                        NULL));

    // 配置AP模式（保持原有配置）
    wifi_config_t ap_config = {
        .ap = {
            .ssid = EXAMPLE_ESP_WIFI_SSID,
            .ssid_len = strlen(EXAMPLE_ESP_WIFI_SSID),
            .channel = EXAMPLE_ESP_WIFI_CHANNEL,
            .password = EXAMPLE_ESP_WIFI_PASS,
            .max_connection = EXAMPLE_MAX_STA_CONN,
#ifdef CONFIG_ESP_WIFI_SOFTAP_SAE_SUPPORT
            .authmode = WIFI_AUTH_WPA2_WPA3_PSK,
            .sae_pwe_h2e = WPA3_SAE_PWE_BOTH,
#else
            .authmode = WIFI_AUTH_WPA2_PSK,
#endif
            .pmf_cfg = {
                .required = false,
                .capable = true
            },
        },
    };
    if (strlen(EXAMPLE_ESP_WIFI_PASS) == 0) {
        ap_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    // 检查STA配置是否有效
    bool sta_enabled = (strlen(cfg.sta_ssid) > 0) && (strlen(cfg.sta_password) > 0);
    
    if (sta_enabled) {
        // 配置STA模式（从配置文件中读取）
        wifi_config_t sta_config = {
            .sta = {
                .ssid = "",
                .password = "",
                .scan_method = WIFI_FAST_SCAN,
                .sort_method = WIFI_CONNECT_AP_BY_SIGNAL,
                .threshold.rssi = -127,
                .threshold.authmode = WIFI_AUTH_OPEN,
            },
        };
        strncpy((char*)sta_config.sta.ssid, cfg.sta_ssid, sizeof(sta_config.sta.ssid));
        strncpy((char*)sta_config.sta.password, cfg.sta_password, sizeof(sta_config.sta.password));
        
        // 设置WiFi模式为AP+STA
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
        
        // 配置STA
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
        
        ESP_LOGI(TAG, "STA enabled, SSID:%s", cfg.sta_ssid);
    } else {
        // 设置WiFi模式为仅AP
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
        ESP_LOGI(TAG, "STA disabled, using AP mode only");
    }
    
    // 配置AP
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta_ap finished. AP SSID:%s password:%s channel:%d",
             EXAMPLE_ESP_WIFI_SSID, EXAMPLE_ESP_WIFI_PASS, EXAMPLE_ESP_WIFI_CHANNEL);
}

static const camera_config_t camera_config = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    //XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_1,  // 使用不同的定时器避免冲突
    .ledc_channel = LEDC_CHANNEL_1,

    .pixel_format = PIXFORMAT_JPEG, //YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_QVGA,    //QQVGA-UXGA, For ESP32, do not use sizes above QVGA when not JPEG. The performance of the ESP32-S series has improved a lot, but JPEG mode always gives better frame rates.
    // 如果数值过小，容易报cam_hal: FB-OVF错误
    // 可以尝试在 menuconfig 中增大 Custom JPEG mode frame size (bytes) 选项的值来增大 jpeg recv buffer 的大小。
    .jpeg_quality = 8, //0-63, for OV series camera sensors, lower number means higher quality
    .fb_count = 2,     //When jpeg mode is used, if fb_count more than one, the driver will work in continuous mode.
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

static esp_err_t init_camera(bool first_call, int ae_level)
{
    if(first_call){
        esp_err_t err = esp_camera_init(&camera_config);
        if (err != ESP_OK)
        {
            ESP_LOGE(TAG, "Camera Init Failed");
            return err;
        }
    }
    
    // 2. 获取传感器对象
    sensor_t *s = esp_camera_sensor_get();

    // 3. 配置各种参数
    s->set_exposure_ctrl(s, 1);  // 启用自动曝光 (0=关闭, 1=开启)
    s->set_ae_level(s, ae_level);// 曝光级别 (-5 ~ +5)
    s->set_vflip(s, 1);// 垂直镜像
    // s->set_wb_mode(s, 4); // 白平衡模式

    // 配置像素格式
    // s->set_pixformat(s, PIXFORMAT_JPEG); // 或者 PIXFORMAT_RGB565
    
    return ESP_OK;
}

// 初始化PWM控制器
static void init_pwm(void)
{
    // 配置定时器
    ledc_timer_config_t ledc_timer = {
        .speed_mode       = LEDC_MODE,
        .duty_resolution  = LEDC_DUTY_RES,
        .timer_num        = LEDC_TIMER,
        .freq_hz          = LEDC_FREQUENCY,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));

    // 配置通道
    ledc_channel_config_t ledc_channel = {
        .speed_mode     = LEDC_MODE,
        .channel        = LEDC_CHANNEL,
        .timer_sel      = LEDC_TIMER,
        .intr_type      = LEDC_INTR_DISABLE,
        .gpio_num       = GPIO_LED,
        .duty           = 0,
        .hpoint         = 0
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ledc_channel));
}

static void update_led_brightness(int led_brightness)
{
    // 默认亮度设置计算方式
    uint32_t max_duty = (1 << LEDC_DUTY_RES) - 1;  // 1023 for 10-bit
    uint32_t duty = max_duty * led_brightness / 100;
    ledc_set_duty(LEDC_MODE, LEDC_CHANNEL, duty);
    ledc_update_duty(LEDC_MODE, LEDC_CHANNEL);
}

static void update_quads_info(void)
{
    get_2x3x3_quads(cfg.color_detect_points, cfg.color_detect_quads);
    // int max_quads_area = 0;
    // for(int i=0; i<18; i++){
    //     int area = calculate_quad_area(cfg.color_detect_quads[i]);
    //     if(area > max_quads_area){
    //         max_quads_area = area;
    //     }
    // }
    // // 预留5%的余量
    // cfg.max_quads_area = (int)(max_quads_area * 1.1f);
    // ESP_LOGI(TAG, "max_quads_area = %d", cfg.max_quads_area);
}

static void init_gpio(void)
{
    const gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GPIO_DBG_LED),  // 选择GPIO引脚
        .mode = GPIO_MODE_OUTPUT,                // 输出模式
        .intr_type = GPIO_INTR_DISABLE,          // 禁用中断
        .pull_down_en = 0,                       // 不启用下拉电阻
        .pull_up_en = 0                          // 不启用上拉电阻
    };
    
    gpio_config(&io_conf);

    const gpio_config_t io_conf_input = {
        .pin_bit_mask = (1ULL << GPIO_KEY),
        .mode = GPIO_MODE_INPUT,
        .intr_type = GPIO_INTR_DISABLE,
        .pull_down_en = 0, 
        .pull_up_en = GPIO_PULLUP_ENABLE
    };
    
    gpio_config(&io_conf_input);
}
static bool check_restore_key_press(void)
{
    if(gpio_get_level(GPIO_KEY) == 0){
        vTaskDelay(pdMS_TO_TICKS(10));
        if(gpio_get_level(GPIO_KEY) == 0){
            return true;
        }
    }
    if(atomic_load(&g_restore)){
        atomic_store(&g_restore, false);
        return true;
    }
    return false;
}


// -------------------------------------- 颜色识别代码 -------------------------------------- 
const uint8_t index_mapping[54] = {
    6,3,0,7,4,1,8,5,2,
    47,50,53,46,49,52,45,48,51,
    17,16,15,14,13,12,11,10,9,
    26,25,24,23,22,21,20,19,18,
    38,41,44,37,40,43,36,39,42,
    27,28,29,30,31,32,33,34,35
};
// -------------------------------------- Web界面部分 -------------------------------------- 
// 添加视频流处理函数
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t res = ESP_OK;
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK){
        return res;
    }

    while(true){
        jpeg_buf jpg_buf;
        if (xQueueReceive(jpeg_queue, &jpg_buf, portMAX_DELAY) != pdPASS){
            continue;
        }
        if(res == ESP_OK){
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if(res == ESP_OK){
            char *part_buf[64];
            size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, jpg_buf.len);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if(res == ESP_OK){
            res = httpd_resp_send_chunk(req, (const char *)jpg_buf.buf, jpg_buf.len);
        }
        free(jpg_buf.buf);
        if(res != ESP_OK){
            break;
        }
    }
    return res;
}

// 只挂载SPIFFS不加载文件
static void mount_spiffs(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true};

    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&conf));
}
// 修改请求处理函数，直接从文件系统读取并发送
static esp_err_t send_html_file(httpd_req_t *req, const char *filename) 
{
    // 打开文件
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        ESP_LOGE(TAG, "Failed to open %s", filename);
        const char *resp = "404 Not Found";
        httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
        return ESP_FAIL;
    }

    // 获取文件大小
    // struct stat st;
    // if (stat(filename, &st)) {
    //     fclose(fp);
    //     ESP_LOGE(TAG, "Failed to stat %s", filename);
    //     const char *resp = "500 Internal Server Error";
    //     httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
    //     return ESP_FAIL;
    // }

    // 设置HTTP响应头
    httpd_resp_set_type(req, "text/html");

    // 分块发送文件内容
    char buffer[256];
    size_t read_bytes;
    while ((read_bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (httpd_resp_send_chunk(req, buffer, read_bytes) != ESP_OK) {
            fclose(fp);
            ESP_LOGE(TAG, "File sending failed");
            return ESP_FAIL;
        }
    }

    // 结束分块传输
    httpd_resp_send_chunk(req, NULL, 0);
    fclose(fp);
    return ESP_OK;
}

static esp_err_t get_req_index_html_handler(httpd_req_t *req)
{
    return send_html_file(req, "/spiffs/index.html");
}

static esp_err_t get_req_editor_html_handler(httpd_req_t *req) {
    return send_html_file(req, "/spiffs/config.html");
}

static esp_err_t get_config_handler(httpd_req_t *req) {
    char filename[50];
    char file_param[20];
    
    if (httpd_req_get_url_query_str(req, filename, sizeof(filename)) != ESP_OK) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    if (httpd_query_key_value(filename, "file", file_param, sizeof(file_param)) != ESP_OK) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    char fullpath[50];
    snprintf(fullpath, sizeof(fullpath), "/spiffs/%s", file_param);
    
    FILE *fp = fopen(fullpath, "r");
    if (!fp) {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    
    // 获取文件大小
    // fseek(fp, 0, SEEK_END);
    // long fsize = ftell(fp);
    // fseek(fp, 0, SEEK_SET);
    
    // 分块发送文件内容
    char buffer[256];
    size_t read_bytes;
    
    httpd_resp_set_type(req, "text/plain");
    
    while ((read_bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (httpd_resp_send_chunk(req, buffer, read_bytes) != ESP_OK) {
            fclose(fp);
            return ESP_FAIL;
        }
    }
    
    fclose(fp);
    httpd_resp_send_chunk(req, NULL, 0);  // 结束分块传输
    return ESP_OK;
}

static esp_err_t send_http_err(httpd_req_t *req, const char *msg)
{
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, msg);
    return ESP_FAIL;
}

static int spiffs_replace_file(const char *tmp_path, const char *dst_path)
{
    char bak_path[96];
    snprintf(bak_path, sizeof(bak_path), "%s.bak", dst_path);

    // 清理旧备份
    remove(bak_path);

    // 先把旧文件挪到备份（若不存在则忽略）
    if (rename(dst_path, bak_path) != 0) {
        if (errno != ENOENT) {
            ESP_LOGE(TAG, "replace: rename old->bak failed (%d): %s", errno, strerror(errno));
            return -1;
        }
    }

    // 再把tmp变成正式文件
    if (rename(tmp_path, dst_path) != 0) {
        ESP_LOGE(TAG, "replace: rename tmp->dst failed (%d): %s", errno, strerror(errno));
        // 回滚
        rename(bak_path, dst_path);
        return -1;
    }

    // 成功后删除备份
    remove(bak_path);
    return 0;
}

static esp_err_t save_config_handler(httpd_req_t *req) {
    char query[64] = {0};
    char file_param[20] = {0};

    // 1) 解析并校验 query
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        return send_http_err(req, "missing query");
    }
    if (httpd_query_key_value(query, "file", file_param, sizeof(file_param)) != ESP_OK) {
        return send_http_err(req, "missing file");
    }

    // 2) 严格校验 body 长度（关键：防空 body 清空文件）
    if (req->content_len <= 0) {
        ESP_LOGE(TAG, "save_config: empty body, file=%s", file_param);
        return send_http_err(req, "empty body");
    }

    // 限制最大大小，防异常请求
    const int MAX_CFG_SIZE = 256 * 1024;
    if (req->content_len > MAX_CFG_SIZE) {
        ESP_LOGE(TAG, "save_config: body too large=%d, file=%s", req->content_len, file_param);
        return send_http_err(req, "content too large");
    }

    char fullpath[64];
    char tmppath[72];
    snprintf(fullpath, sizeof(fullpath), "/spiffs/%s", file_param);
    snprintf(tmppath, sizeof(tmppath), "/spiffs/%s.tmp", file_param);

    // 3) 分块接收 + 写入临时文件（不占大块堆内存）
    FILE *fp = fopen(tmppath, "wb");
    if (!fp) {
        ESP_LOGE(TAG, "save_config: open tmp failed: %s", tmppath);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    int remaining = req->content_len;
    int total_received = 0;
    char buf[1024];

    while (remaining > 0) {
        int to_read = remaining > (int)sizeof(buf) ? (int)sizeof(buf) : remaining;
        int ret = httpd_req_recv(req, buf, to_read);

        if (ret == HTTPD_SOCK_ERR_TIMEOUT) {
            continue; // 重试
        }
        if (ret <= 0) {
            ESP_LOGE(TAG, "save_config: recv failed ret=%d, recv=%d/%d",
                     ret, total_received, req->content_len);
            fclose(fp);
            remove(tmppath);
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }

        size_t written = fwrite(buf, 1, ret, fp);
        if (written != (size_t)ret) {
            ESP_LOGE(TAG, "save_config: fwrite failed, written=%u ret=%d",
                     (unsigned)written, ret);
            fclose(fp);
            remove(tmppath);
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }

        total_received += ret;
        remaining -= ret;
    }

    fflush(fp);
    fclose(fp);

    if (total_received != req->content_len || total_received <= 0) {
        ESP_LOGE(TAG, "save_config: size mismatch recv=%d, content_len=%d",
                 total_received, req->content_len);
        remove(tmppath);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // 4) 原子替换（关键：不会把原文件直接截断成空）
    if (spiffs_replace_file(tmppath, fullpath) != 0) {
        ESP_LOGE(TAG, "save_config: rename failed %s -> %s", tmppath, fullpath);
        remove(tmppath);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "save_config: success file=%s size=%d", file_param, total_received);

    // 5) 重新加载配置
    if (strcmp(file_param, "main.cfg") == 0) {
        if (load_main_cfg(fullpath) == 0) {
            update_led_brightness(cfg.led_brightness);
            update_quads_info();
        } else {
            ESP_LOGE(TAG, "save_config: reload main.cfg failed");
        }
    } else { // motion.cfg
        if (load_motion_cfg(fullpath) == 0) {
            init_time_table(&get_time_motions_str);
            update_camera_flip_time();
        } else {
            ESP_LOGE(TAG, "save_config: reload motion.cfg failed");
        }
    }

    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

static esp_err_t download_config_handler(httpd_req_t *req) {
    char filename[50];
    if (httpd_req_get_url_query_str(req, filename, sizeof(filename)) != ESP_OK) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    char file_param[20];
    if (httpd_query_key_value(filename, "file", file_param, sizeof(file_param)) != ESP_OK) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    char fullpath[50];
    snprintf(fullpath, sizeof(fullpath), "/spiffs/%s", file_param);

    FILE *fp = fopen(fullpath, "rb");
    if (!fp) {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    // fseek(fp, 0, SEEK_END);
    // long fsize = ftell(fp);
    // fseek(fp, 0, SEEK_SET);

    // 设置下载头
    char header[100];
    snprintf(header, sizeof(header), "attachment; filename=\"%s\"", file_param);
    httpd_resp_set_hdr(req, "Content-Disposition", header);
    httpd_resp_set_type(req, "application/octet-stream");

    // 分块发送文件
    char buffer[256];
    size_t read_bytes;
    while ((read_bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (httpd_resp_send_chunk(req, buffer, read_bytes) != ESP_OK) {
            fclose(fp);
            return ESP_FAIL;
        }
    }
    fclose(fp);

    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

static esp_err_t handle_ws_req(httpd_req_t *req)
{
    if (req->method == HTTP_GET)
    {
        ESP_LOGI(TAG, "Handshake done, the new connection was opened");
        return ESP_OK;
    }

    httpd_ws_frame_t ws_pkt;
    uint8_t *buf = NULL;
    memset(&ws_pkt, 0, sizeof(httpd_ws_frame_t));
    ws_pkt.type = HTTPD_WS_TYPE_TEXT;
    esp_err_t ret = httpd_ws_recv_frame(req, &ws_pkt, 0);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "httpd_ws_recv_frame failed to get frame len with %d", ret);
        return ret;
    }

    if (ws_pkt.len)
    {
        buf = calloc(1, ws_pkt.len + 1);
        if (buf == NULL)
        {
            ESP_LOGE(TAG, "Failed to calloc memory for buf");
            return ESP_ERR_NO_MEM;
        }
        ws_pkt.payload = buf;
        ret = httpd_ws_recv_frame(req, &ws_pkt, ws_pkt.len);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "httpd_ws_recv_frame failed with %d", ret);
            free(buf);
            return ret;
        }
        // ESP_LOGI(TAG, "Got packet with message: %s", ws_pkt.payload);
    }
    // ESP_LOGI(TAG, "frame len is %d", ws_pkt.len);

    if (ws_pkt.type == HTTPD_WS_TYPE_TEXT)
    {
        if (strcmp((char *)ws_pkt.payload, "get_brightness") == 0) {
            char buff[32];
            sprintf(buff, "brightness:%d", cfg.led_brightness);
            httpd_ws_frame_t ws_pkt_resp = {
                .payload = (uint8_t *)buff,
                .len = strlen(buff),
                .type = HTTPD_WS_TYPE_TEXT
            };
            httpd_ws_send_frame(req, &ws_pkt_resp);
        }
        // 处理亮度控制消息
        else if (strncmp((char *)ws_pkt.payload, "brightness:", 11) == 0) {
            cfg.led_brightness = atoi((char *)ws_pkt.payload + 11);
            if (cfg.led_brightness < 0) cfg.led_brightness = 0;
            if (cfg.led_brightness > 100) cfg.led_brightness = 100;
            // 设置PWM占空比
            update_led_brightness(cfg.led_brightness);
            char buff[32];
            sprintf(buff, "brightness:%d", cfg.led_brightness);
            httpd_ws_frame_t ws_pkt_resp = {
                .payload = (uint8_t *)buff,
                .len = strlen(buff),
                .type = HTTPD_WS_TYPE_TEXT
            };
            httpd_ws_send_frame(req, &ws_pkt_resp);
        }
        // 保存坐标
        else if (strcmp((char *)ws_pkt.payload, "save") == 0) {
            save_main_cfg(MAIN_CFG_PATH);
        } 
        // 处理坐标更新消息
        else if (strncmp((char *)ws_pkt.payload, "update_pos:", 11) == 0) {
            char *pos_str = (char *)ws_pkt.payload + 11;
            // ESP_LOGI(TAG, "Received positions: %s", pos_str);
            
            // 解析坐标字符串 (格式: (x1,y1),(x2,y2),...,(x6,y6))
            char *token = strtok(pos_str, "(),");
            for (int i = 0; i < 6 && token != NULL; i++) {
                // 读取x坐标
                cfg.color_detect_points[i].x = atoi(token);
                token = strtok(NULL, "(),");
                // 读取y坐标
                if (token != NULL) {
                    cfg.color_detect_points[i].y = atoi(token);
                    token = strtok(NULL, "(),");
                }
                // ESP_LOGI(TAG, "Point %d: (%d, %d)", i, 
                //          color_detect_points[i].x, color_detect_points[i].y);
            }
            
            // 更新颜色检测四边形
            update_quads_info();
        }
        // 处理获取坐标请求
        else if (strcmp((char *)ws_pkt.payload, "get_pos") == 0) {
            // 格式化坐标字符串 (格式: (x1,y1),(x2,y2),...,(x6,y6))
            char pos_str[100];
            snprintf(pos_str, sizeof(pos_str), 
                     "positions:(%hd,%hd),(%hd,%hd),(%hd,%hd),(%hd,%hd),(%hd,%hd),(%hd,%hd)",
                     cfg.color_detect_points[0].x, cfg.color_detect_points[0].y,
                     cfg.color_detect_points[1].x, cfg.color_detect_points[1].y,
                     cfg.color_detect_points[2].x, cfg.color_detect_points[2].y,
                     cfg.color_detect_points[3].x, cfg.color_detect_points[3].y,
                     cfg.color_detect_points[4].x, cfg.color_detect_points[4].y,
                     cfg.color_detect_points[5].x, cfg.color_detect_points[5].y);
            
            httpd_ws_frame_t ws_pkt_resp = {
                .payload = (uint8_t *)pos_str,
                .len = strlen(pos_str),
                .type = HTTPD_WS_TYPE_TEXT
            };
            httpd_ws_send_frame(req, &ws_pkt_resp);
        }
        // 处理获取颜色请求，顺便返回视觉模块状态（一直更新）和电机角度信息（如果需要更新）
        // status:正在解魔方;angles:117714, 213808, -22894, -47440;colors:6,192,75,0,9,192,57,128,60,128,36,160,34,160,36,160,105,180,0,192,0,192,60,0,0,192,0,192,6,192,0,192,33,160,33,160,6,192,105,192,60,0,6,192,34,160,60,0,51,128,60,0,60,0,0,192,106,178,54,128,60,128,107,192,9,192,9,192,3,192,106,160,36,160,3,192,59,135,76,0,9,192,105,192,60,128,36,128,36,128,108,192,60,128,105,192,58,128,64,0,6,192,3,192,108,160,33,0,LBLUUFFFDRRBRRLRFFLDBLFBUBBRDUUDLLRDFRUBLDUFFDUDUBLRDB
        else if (strcmp((char *)ws_pkt.payload, "get_colors") == 0) {
            // 获取当前状态
            static const char *state_str[] = {
                "初始化",
                "背景色采集",
                "检测魔方放置",
                "图像稳定中",
                "捕获第1个面",
                "翻转到第2个面",
                "捕获第2个面",
                "翻转到第3个面",
                "捕获第3个面",
                "正在解魔方",
                "检测魔方移除"
            };
            int state = atomic_load(&g_camera_state);
            float time_cost = atomic_load(&g_time_cost_us) / 1000000.0f;
            // 估算还原进度
            int percent = 0;
            if(state <= STATE_DETECTING){
                percent = 0;
            }
            else if(state == STATE_DETECTING_REMOVE){
                percent = 100;
            }else{
                percent = roundf(1000.0f * time_cost / ((g_estimate_time.other + g_estimate_time.camera_flip * 2 + 
                           g_estimate_time.motion_sequence)) * 100.0f);
                if(percent >= 100){
                    percent = 100;
                }
            }

            int angle[4];
            int32_t *angle_encoder = get_g_angle();
            for(int i=0; i<4 ;i++){
                angle[i] = angle_encoder[i] % 16384;
                if(angle[i] < 0){
                    angle[i] += 16384;
                }
            } 
            // 构建响应字符串
            char response[1024];
            snprintf(response, sizeof(response), "status:%s(计时: %.2fs, %d%%);angles:%d, %d, %d, %d;colors:", 
                    state_str[state], time_cost, percent, angle[0], angle[1], angle[2], angle[3]);
            
            // 添加颜色数据
            char temp[20];
            for (int i = 0; i < 54; i++) {
                if (i > 0) {
                    strcat(response, ",");
                }
                snprintf(temp, sizeof(temp), "%d,%d,%d", g_color_data[i][0], g_color_data[i][1], g_color_data[i][2]);
                strcat(response, temp);
            }
            strcat(response, ",");
            strcat(response, g_cube_str);
            
            httpd_ws_frame_t ws_pkt_resp = {
                .payload = (uint8_t *)response,
                .len = strlen(response),
                .type = HTTPD_WS_TYPE_TEXT
            };
            httpd_ws_send_frame(req, &ws_pkt_resp);
            // ESP_LOGI(TAG,"%s",response);
        }
        // 测试运动控制
        else if (strcmp((char *)ws_pkt.payload, "disable_all") == 0) {
            motion_cmd_t cmd = CMD_DISABLE_ALL;
            xQueueSend(motion_cmd_queue, &cmd, 0);
        }
        else if (strcmp((char *)ws_pkt.payload, "enable_all") == 0) {
            motion_cmd_t cmd = CMD_ENABLE_ALL;
            xQueueSend(motion_cmd_queue, &cmd, 0);
        }
        else if (strcmp((char *)ws_pkt.payload, "clamp_cube") == 0) {
            motion_cmd_t cmd = CMD_CLAMP_CUBE;
            xQueueSend(motion_cmd_queue, &cmd, 0);
        }
        else if (strcmp((char *)ws_pkt.payload, "release_cube") == 0) {
            motion_cmd_t cmd = CMD_RELEASE_CUBE;
            xQueueSend(motion_cmd_queue, &cmd, 0);
        }
        else if (strcmp((char *)ws_pkt.payload, "go_home") == 0) {
            motion_cmd_t cmd = CMD_GO_HOME;
            xQueueSend(motion_cmd_queue, &cmd, 0);
        }
        else if (strcmp((char *)ws_pkt.payload, "restore") == 0) {
            if(cfg.auto_cube_detect == CUBE_DETECT_MANUAL &&
               atomic_load(&g_camera_state) == STATE_DETECTING){
                atomic_store(&g_restore, true);
            }
        }
        else if (strncmp((char *)ws_pkt.payload, "scramble:", 9) == 0) {
            char *sequence = (char *)ws_pkt.payload + 9;
            size_t len = strlen(sequence);
            if(len == 0){
                strcpy(cfg.scramble_sequence, "RANDOM");
            }
            else if (len < SCRAMBLE_SEQUENCE_LEN - 1) {
                strncpy(cfg.scramble_sequence, sequence, len);
                cfg.scramble_sequence[len] = '\0';
            } else {
                ESP_LOGE(TAG, "打乱序列过长");
            }
            save_main_cfg(MAIN_CFG_PATH);
        }else if (strncmp((char *)ws_pkt.payload, "get_scramble", 12) == 0) {
            char response[256];
            snprintf(response, sizeof(response), "scramble_sequence:%s", cfg.scramble_sequence);
            httpd_ws_frame_t ws_pkt_resp = {
                .payload = (uint8_t *)response,
                .len = strlen(response),
                .type = HTTPD_WS_TYPE_TEXT
            };
            httpd_ws_send_frame(req, &ws_pkt_resp);
        }
        else if (strncmp((char *)ws_pkt.payload, "motions:", 8) == 0) {
            char *sequence = (char *)ws_pkt.payload + 8;
            size_t len = strlen(sequence);
            if (len < MOTION_SEQUENCE_LEN) {
                strncpy(g_motion_sequence, sequence, len);
                g_motion_sequence[len] = '\0';
                g_estimate_time.motion_sequence = 0;
                motion_cmd_t cmd = CMD_MOTIONS;
                xQueueSend(motion_cmd_queue, &cmd, 0);
            } else {
                ESP_LOGE(TAG, "动作序列过长");
            }
        }
        else{
            ESP_LOGW(TAG, "Unknown command %s.", ws_pkt.payload);
        }
    }
    free(buf);
    return ESP_OK;
}

void setup_servers(void) {
    // 主服务器配置 (处理WebSocket等)
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.stack_size = 8192; // 默认4096
    config.server_port = 80;
    config.max_uri_handlers = 16;  // 增加最大URI处理程序数量
    
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        ESP_LOGI(TAG, "Main HTTP server started on port %d", config.server_port);
        
        // 注册WebSocket和网页请求
        httpd_uri_t uri_get = {
            .uri = "/",
            .method = HTTP_GET,
            .handler = get_req_index_html_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(camera_httpd, &uri_get);
        
        httpd_uri_t ws = {
            .uri = "/ws",
            .method = HTTP_GET,
            .handler = handle_ws_req,
            .user_ctx = NULL,
            .is_websocket = true
        };
        httpd_register_uri_handler(camera_httpd, &ws);

        // 配置文件编辑器
        httpd_uri_t cfg_editor = {
            .uri = "/config.html",
            .method = HTTP_GET,
            .handler = get_req_editor_html_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(camera_httpd, &cfg_editor);
        
        // 获取配置文件内容
        httpd_uri_t cfg_get = {
            .uri = "/cfg/get",
            .method = HTTP_GET,
            .handler = get_config_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(camera_httpd, &cfg_get);
        
        // 保存配置文件
        httpd_uri_t cfg_save = {
            .uri = "/cfg/save",
            .method = HTTP_POST,
            .handler = save_config_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(camera_httpd, &cfg_save);
        
        // 下载配置文件
        httpd_uri_t cfg_download = {
            .uri = "/cfg/download",
            .method = HTTP_GET,
            .handler = download_config_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(camera_httpd, &cfg_download);

    } else {
        ESP_LOGE(TAG, "Failed to start main HTTP server on port %d", config.server_port);
    }
    
    // 流服务器配置 (专门处理视频流)
    httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
    stream_config.server_port = 81;  // 使用不同端口
    stream_config.ctrl_port += 1;    // 增加控制端口
    
    if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
        ESP_LOGI(TAG, "Stream HTTP server started on port %d", stream_config.server_port);
        
        httpd_uri_t stream_uri = {
            .uri = "/stream",
            .method = HTTP_GET,
            .handler = stream_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(stream_httpd, &stream_uri);
    } else {
        ESP_LOGE(TAG, "Failed to start stream HTTP server on port %d", stream_config.server_port);
    }
}
// -------------------------------------- 电机控制 -------------------------------------- 
void motion_task(void *arg) 
{
    bool motor_enable = false;
    bool motor_ready = false;
    // 检查4个电机控制器的状态
    for(int i=0; i<16; i++){
        if(motion_self_test() == 0){
            motor_ready = true;
            break;
        }else{
            ESP_LOGE(TAG, "电机状态检查失败，0.5秒后重试。");
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
    if(motor_ready){
        motion_reset_all();
        for(int i=0;i<3;i++){
            int succ = cmd_zero();
            if(succ == 0){
                motor_enable = true;
                two_finger_init();
                break;
            }else{
                disable_all();
                ESP_LOGE(TAG, "回零点失败");
                vTaskDelay(pdMS_TO_TICKS(1000));
            }
        }
    }

    while (1)
    {
        atomic_store(&g_motor_enable, motor_enable);
        motion_cmd_t cmd;
        if (xQueueReceive(motion_cmd_queue, &cmd, pdMS_TO_TICKS(100)) != pdPASS){
            // 空闲时自动查询电机角度
            if(motor_ready){
                print_pos();
            }
            continue;
        }
        // 如果电机状态异常，不执行任何指令
        if(!motor_ready){
            continue;
        }
        switch (cmd) {
            case CMD_DISABLE_ALL: 
                disable_all();
                motor_enable = false;
                break;
            case CMD_ENABLE_ALL:
                enable_all();
                motor_enable = true;
                break;
            case CMD_QUERY_ANGLES:
                break;
            case CMD_CLAMP_CUBE:
                // 实现夹紧魔方逻辑
                if(motor_enable){
                    two_finger_clamp();
                    atomic_store(&g_flip_completed, true);
                }
                break;
            case CMD_RELEASE_CUBE:
                // 实现释放魔方逻辑
                if(motor_enable){
                    two_finger_init();
                    atomic_store(&g_flip_completed, true);
                }
                break;
            case CMD_GO_HOME:
                // 实现回零位逻辑
                motion_reset_all();
                if (cmd_zero() == 0) {
                    two_finger_init();
                    ESP_LOGI(TAG, "回零点成功");
                } else {
                    disable_all();
                    ESP_LOGE(TAG, "回零点失败");
                }
                atomic_store(&g_flip_completed, true);
                atomic_store(&g_motor_error, false);// 可通过回零点清除故障标志
                motor_enable = true;
                break;
            case CMD_MOTIONS: 
                if(motor_enable){
                    two_finger_clamp();
                    if(motions(g_motion_sequence) != 0){
                        atomic_store(&g_motor_error, true);
                        disable_all();
                    }else{
                        two_finger_init();
                    }
                    atomic_store(&g_flip_completed, true);
                }
                break;
            case CMD_FLIP_TO_FACE_2:
                if(motor_enable){
                    two_finger_clamp();
                    // 优化前 R1 L*F R0   L1 R+F L0   R2 R+N R0
                    // 优化后 L1 R-F L0   R2 R+N L*F R0
                    if(motions(CAMERA_FLIP_SEQUANCE_1) != 0){
                        atomic_store(&g_motor_error, true);
                    }
                    atomic_store(&g_flip_completed, true);
                }
                break;
            case CMD_FLIP_TO_FACE_3:
                if(motor_enable){
                    // 优化前 L1 R*F L0   R1 L+F R0   L2 L+N L0
                    // 优化后 R1 L-F R0   L2 L+N R*F L0
                    if(motions(CAMERA_FLIP_SEQUANCE_2) != 0){
                        atomic_store(&g_motor_error, true);
                    }
                    atomic_store(&g_flip_completed, true);
                }
                break;
            default:
                ESP_LOGW(TAG, "未知指令: %d", cmd);
                break;
        }
    }
}

// -------------------------------------- 摄像头图像采集与魔方求解程序 -------------------------------------- 
// 解压JPEG
static jpeg_error_t jpeg_decode_one_frame(camera_fb_t *fb, uint8_t **output_buf)
{
    jpeg_error_t ret = JPEG_ERR_OK;
    jpeg_dec_io_t *jpeg_io = NULL;
    jpeg_dec_header_info_t *out_info = NULL;

    // Generate default configuration
    jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
    config.output_type = JPEG_PIXEL_FORMAT_RGB565_LE;
    config.rotate = JPEG_ROTATE_0D;

    // Create jpeg_dec handle
    jpeg_dec_handle_t jpeg_dec = NULL;
    ret = jpeg_dec_open(&config, &jpeg_dec);
    if (ret != JPEG_ERR_OK)
    {
        return ret;
    }

    // Create io_callback handle
    jpeg_io = calloc(1, sizeof(jpeg_dec_io_t));
    if (jpeg_io == NULL)
    {
        ret = JPEG_ERR_NO_MEM;
        goto jpeg_dec_failed;
    }

    // Create out_info handle
    out_info = calloc(1, sizeof(jpeg_dec_header_info_t));
    if (out_info == NULL)
    {
        ret = JPEG_ERR_NO_MEM;
        goto jpeg_dec_failed;
    }

    // Set input buffer and buffer len to io_callback
    jpeg_io->inbuf = fb->buf;
    jpeg_io->inbuf_len = fb->len;

    // Parse jpeg picture header and get picture for user and decoder
    ret = jpeg_dec_parse_header(jpeg_dec, jpeg_io, out_info);
    if (ret != JPEG_ERR_OK) {
        goto jpeg_dec_failed;
    }

    int out_len = out_info->width * out_info->height * 2;
    // 如果没有分配输出内存，进行分配
    if(*output_buf == NULL){
        *output_buf = jpeg_calloc_align(out_len, 16);
        ESP_LOGI(TAG, "jpeg_calloc_align, size = %dKB", out_len/1024);
    }
    if (*output_buf == NULL) {
        ret = JPEG_ERR_NO_MEM;
        goto jpeg_dec_failed;
    }
    jpeg_io->outbuf = *output_buf;

    // Start decode jpeg
    ret = jpeg_dec_process(jpeg_dec, jpeg_io);
    if (ret != JPEG_ERR_OK) {
        goto jpeg_dec_failed;
    }

    // Decoder deinitialize
jpeg_dec_failed:
    jpeg_dec_close(jpeg_dec);
    if (jpeg_io) {
        free(jpeg_io);
    }
    if (out_info) {
        free(out_info);
    }
    return ret;
}

static void get_color(uint8_t *img, int width, int height, uint8_t output_buffer[18][3])
{
    for (int i = 0; i < 18; i++) {
        uint8_t dominant_h, dominant_s, dominant_v;
        // 使用avg_mode指定的模式提取18个色块的颜色信息
        extract_dominant_color_from_quad(cfg.color_detect_quads[i], img, width, height, 
                &dominant_h, &dominant_s, &dominant_v);
            
        output_buffer[i][0] = dominant_h;
        output_buffer[i][1] = dominant_s;
        output_buffer[i][2] = dominant_v;
    }
}
// RGB565转grey
static inline uint8_t rgb565_to_grey(uint16_t pixel) 
{
    // 提取RGB分量
    uint8_t r5 = (pixel >> 11) & 0x1F;  // 高5位是红色 (0-31)
    uint8_t g6 = (pixel >> 5)  & 0x3F;  // 中间6位是绿色 (0-63)
    uint8_t b5 = pixel & 0x1F;          // 低5位是蓝色 (0-31)

    // 整数运算：灰度 = (154 * R5 + 150 * G6 + 58 * B5) >> 6
    // 推导：标准公式转换为 5/6 位分量的系数缩放
    uint32_t sum = (r5 * 154) + (g6 * 150) + (b5 * 58);
    return (uint8_t)(sum >> 6);
} 

// 存放颜色数据，用于实时预览或者魔方解算
typedef enum{
    FACE_1 = 0,
    FACE_2 = 18,
    FACE_3 = 36
}cube_face_t;
static void update_g_color_data(cube_face_t face, const uint8_t color_data[18][3])
{
    // 色块映射表
    const uint8_t index_mapping[54] = {
        6, 3, 0, 7, 4, 1, 8, 5, 2,
        47,50,53,46,49,52,45,48,51,
        17,16,15,14,13,12,11,10,9,
        26,25,24,23,22,21,20,19,18,
        38,41,44,37,40,43,36,39,42,
        27,28,29,30,31,32,33,34,35
    };
    for (int i = 0; i < 18; i++) {
        // 查询存放的位置
        int index = index_mapping[i + (int)face];
        // 更新全局变量g_color_data
        g_color_data[index][0] = color_data[i][0];
        g_color_data[index][1] = color_data[i][1];
        g_color_data[index][2] = color_data[i][2];
    }
}

static int run_solve(const char facelets[55], char *motion_sequence, int motion_sequence_len)
{
    // 魔方求解
    int64_t time_start = esp_timer_get_time();
    // 分配变量
    Search *search = malloc(sizeof(Search));         // 用于魔方求解的结构体，约3KB
    char result[SCRAMBLE_SEQUENCE_LEN] = {0};        // 存储魔方还原公式，128B
    int try_count = 0;                               // 调用SearchNext的计数
    int shortest_try_count = 0;                      // 最短机械步骤出现在哪一次SearchNext调用
    int total_time = 0;                              // 预估的机械步骤耗时
    char *motion_temp = malloc(motion_sequence_len); // 暂存的机械步骤，1KB
    int length = 0;                                  // 魔方还原公式的长度
    memset(search, 0, sizeof(Search));

    // 搜索第一个解法
    length = SearchSolution(search, result, facelets, 22, 500, 1);
    if(length >= 0){
        ESP_LOGI(TAG, "SearchSolution: %s (length = %d)", result, length);
        total_time = solution_to_motion("RU", result, motion_sequence);
        // 如果剩余时间充足，尝试寻找更短的解法
        while (1) {
            int elapsed_us = esp_timer_get_time() - time_start;
            if (elapsed_us >= 150 * 1000) {
                break;  // 超过最大耗时(150ms)则退出
            }
            int ret = SearchNext(search, result, 1, 0);
            try_count++;
            if(ret >= 0){
                ESP_LOGI(TAG, "SearchNext try_count=%d: %s (length = %d)", try_count, result, ret);
                int new_time = solution_to_motion("RU", result, motion_temp);
                length = ret;
                // 如果找到的新步骤，总耗时更短(魔方还原公式length短的解法，total_time不一定更短)
                if(new_time <= total_time){
                    total_time = new_time;
                    strncpy(motion_sequence, motion_temp, motion_sequence_len - 1);
                    motion_sequence[motion_sequence_len - 1] = '\0';
                    shortest_try_count = try_count;
                }
            }
        }
    }else{
        motion_sequence[0] = '\0';
        ESP_LOGE(TAG, "不可解的魔方");
    }

    free(search);
    free(motion_temp);

    float solve_time = (esp_timer_get_time() - time_start) / 1000.0f;
    ESP_LOGI(TAG, "机械步骤: %s (try_count=%d, total_time=%dms)", motion_sequence, shortest_try_count ,total_time);
    ESP_LOGI(TAG, "计算求解方式耗时%.2fms", solve_time);
    // return length;
    return total_time;
}

// 魔方解算测试程序
static int run_solve_test(const char *testcase)
{
    FILE *f = fopen(testcase, "r");
    char line_buffer[80];
    int test_case_count = 0;
    if(f != NULL){
        int min = INT_MAX, max=0, avg=0;
        for(int i=0;;i++){ 
            char *line = fgets(line_buffer,79,f);
            if(line == 0){
                break;
            }
            if(strlen(line_buffer) < 54){
                continue;
            }
            line_buffer[54] = '\0';
            ESP_LOGI(TAG, "测试用例%d: %s",i, line_buffer);
            test_case_count++;
            // 测试魔方求解
            int len = run_solve(line_buffer, g_motion_sequence, sizeof(g_motion_sequence));

            if(len > max){
                max = len;
            }
            if(len < min){
                min = len;
            }
            avg += len;
            // 短暂释放CPU，可调度其他低优先级任务
            vTaskDelay(1);
        }
        fclose(f);
        ESP_LOGI(TAG, "min=%d, max=%d, avg=%.2f, test_case_count=%d\n",
                 min, max, avg / (float)test_case_count, test_case_count);
    }else{
        ESP_LOGE(TAG, "can not open %s\n",testcase);
    }
    return 0;
}

// 辅助函数：获取面的三维向量表示
static void get_face_vector(char face, int *vec) {
    switch(face) {
        case 'U': vec[0] = 0;  vec[1] = 1;  vec[2] = 0;  break;  // 上
        case 'D': vec[0] = 0;  vec[1] = -1; vec[2] = 0;  break;  // 下
        case 'F': vec[0] = 0;  vec[1] = 0;  vec[2] = 1;  break;  // 前
        case 'B': vec[0] = 0;  vec[1] = 0;  vec[2] = -1; break;  // 后
        case 'L': vec[0] = -1; vec[1] = 0;  vec[2] = 0;  break;  // 左
        case 'R': vec[0] = 1;  vec[1] = 0;  vec[2] = 0;  break;  // 右
        default:  vec[0] = 0;  vec[1] = 0;  vec[2] = 0;  break;  // 无效
    }
}

// 辅助函数：计算两个向量的叉积
static void cross_product(int *a, int *b, int *result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// 辅助函数：将向量转换为面对应的字符
static char vector_to_face(int *vec) {
    if (vec[0] == 1  && vec[1] == 0  && vec[2] == 0)  return 'R';
    if (vec[0] == -1 && vec[1] == 0  && vec[2] == 0)  return 'L';
    if (vec[0] == 0  && vec[1] == 1  && vec[2] == 0)  return 'U';
    if (vec[0] == 0  && vec[1] == -1 && vec[2] == 0)  return 'D';
    if (vec[0] == 0  && vec[1] == 0  && vec[2] == 1)  return 'F';
    if (vec[0] == 0  && vec[1] == 0  && vec[2] == -1) return 'B';
    return '?';  // 无效向量
}

// 辅助函数：替换解决方案中的面标签
static void transform_solution(const char *src, char *dest, const char *face_map) {
    int i = 0, j = 0;
    while (src[i] != '\0') {
        if (strchr("URFDLB", src[i])) {
            char c = src[i];
            dest[j++] = face_map[c - 'A'];  // 替换面标签
            i++;
        } else {
            dest[j++] = src[i++];  // 直接复制非面字符
        }
    }
    dest[j] = '\0';  // 结束字符串
}

static int run_scramble(char *sequence, char *fixed_scramble, char white_green_pos[3]) {
    // 白中心块：white_green_pos[0]
    // 绿中心块：white_green_pos[1]
    int64_t time_start = esp_timer_get_time();
    sequence[0] = '\0';
    int total_time = 0;

    // 1. 初始化面映射表 - 修改为26个元素
    char face_map[26] = {0}; 
    const char* faces = "URFDLB";
    for (const char *p = faces; *p; p++) {
        face_map[*p - 'A'] = *p;
    }

    // 2. 仅当输入有效时计算新映射
    char U0 = white_green_pos[0];
    char F0 = white_green_pos[1];
    if (strchr(faces, U0) && strchr(faces, F0) && U0 != F0) {
        int U0_vec[3], F0_vec[3];
        get_face_vector(U0, U0_vec);
        get_face_vector(F0, F0_vec);
        
        // 计算右面向量 (U0 × F0)
        int x_vec[3];
        cross_product(U0_vec, F0_vec, x_vec);
        
        // 检查叉积有效性（非零向量）
        if (x_vec[0] != 0 || x_vec[1] != 0 || x_vec[2] != 0) {
            char R0 = vector_to_face(x_vec);
            if (R0 != '?') {
                // 计算其他面的映射
                face_map['U' - 'A'] = U0;
                face_map['F' - 'A'] = F0;
                face_map['R' - 'A'] = R0;
                
                int neg_U0_vec[3] = { -U0_vec[0], -U0_vec[1], -U0_vec[2] };
                face_map['D' - 'A'] = vector_to_face(neg_U0_vec);
                
                int neg_F0_vec[3] = { -F0_vec[0], -F0_vec[1], -F0_vec[2] };
                face_map['B' - 'A'] = vector_to_face(neg_F0_vec);
                
                int R0_vec[3];
                get_face_vector(R0, R0_vec);
                int neg_R0_vec[3] = { -R0_vec[0], -R0_vec[1], -R0_vec[2] };
                face_map['L' - 'A'] = vector_to_face(neg_R0_vec);
            }
        }
    }

    // 3. 处理随机或固定打乱
    if (strstr(fixed_scramble, "RANDOM")) {
        // 产生随机打乱的魔方
        char random_cube[55];
        scramble(random_cube);
        ESP_LOGI(TAG, "随机魔方: %s", random_cube);
        // 计算打乱公式
        char solution[SCRAMBLE_SEQUENCE_LEN] = {0};
        Search *search = (Search *)malloc(sizeof(Search));
        memset(search, 0, sizeof(Search));
        int length = SearchSolution(search, solution, random_cube, 22, 10000, 20);
        free(search);
        // 计算打乱机械步骤
        if (length >= 0) {
            char solution_rev[SCRAMBLE_SEQUENCE_LEN] = {0};
            reverse_solution(solution, solution_rev);
            ESP_LOGI(TAG, "随机打乱公式: %s", solution_rev);
            
            // 转换解决方案
            char transformed_solution[SCRAMBLE_SEQUENCE_LEN] = {0};
            transform_solution(solution_rev, transformed_solution, face_map);
            total_time = solution_to_motion("RU", transformed_solution, sequence);
        }
    } else {
        ESP_LOGI(TAG, "固定打乱公式: %s", fixed_scramble);
        
        // 转换固定打乱公式
        char transformed_scramble[SCRAMBLE_SEQUENCE_LEN] = {0};
        transform_solution(fixed_scramble, transformed_scramble, face_map);
        total_time = solution_to_motion("RU", transformed_scramble, sequence);
    }

    // 4. 性能日志
    int64_t time_end = esp_timer_get_time();
    float solve_time = (time_end - time_start) / 1000.0f;
    ESP_LOGI(TAG, "机械步骤: %s (%dms)", sequence, total_time);
    ESP_LOGI(TAG, "计算求解方式耗时%.2fms", solve_time);
    return total_time;
}

// 上位机指令接收函数
// 使用格式: #<动作序列>(<时间>)
static int receive_host_command(char *buf, int *sequence_time, int buf_size)
{
    uint8_t read_buffer[256];             // 批量读取缓冲区
    bool found_start = false;             // 是否找到起始符'#'
    int output_index = 0;                 // 输出缓冲区索引
    const int64_t RECV_TIMEOUT = 5000000; // 5秒超时
    int64_t recv_start = esp_timer_get_time();
    
    // 循环读取上位机数据
    while (true) {
        // 检查超时
        if (esp_timer_get_time() - recv_start > RECV_TIMEOUT) {
            ESP_LOGE(TAG, "上位机响应超时");
            return -1;
        }
        
        // 批量读取数据（非阻塞）
        int bytes_read = usb_serial_jtag_read_bytes(read_buffer, sizeof(read_buffer), 0);
        if (bytes_read > 0) {
            // 处理本次读取的数据块
            for (int i = 0; i < bytes_read; i++) {
                uint8_t ch = read_buffer[i];
                
                // 忽略所有起始符前的数据
                if (ch == '#') {
                    found_start = true;
                    output_index = 0; // 重置输出缓冲区
                    continue;
                }
                if(found_start){
                    // 处理起始符后的数据
                    if (ch == ')') {
                        // 找到完整指令，立即终止字符串并返回
                        buf[output_index] = '\0';
                        
                        // 查找时间参数
                        char *time_start = strrchr(buf, '(');
                        if (!time_start) {
                            ESP_LOGE(TAG, "无效的时间参数位置");
                            return -1;
                        }

                        *sequence_time = atoi(time_start + 1);
                        *time_start = '\0'; // 分离动作序列
                        
                        ESP_LOGI(TAG, "收到上位机指令: %s (%dms)", buf, *sequence_time);
                        return 0; // 找到完整指令，立即返回
                    }
                    else if (output_index < buf_size - 1) {
                        // 安全存储有效字符
                        buf[output_index++] = ch;
                    }
                    else {
                        // 缓冲区已满但未找到结束符
                        ESP_LOGE(TAG, "指令过长超出缓冲区限制");
                        return -1;
                    }
                }
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

static void camera_task(void *arg) 
{
    int ae_level = cfg.camera_ae_level;
    camera_stat_t state = STATE_INIT;
    int frame_counter = 0;
    uint8_t color_data_background[18][3] = {0}; // HSV背景颜色，用于检测魔方是否存在
    uint8_t color_data_prev[18][3] = {0};       // HSV上一帧颜色 
    uint8_t color_data_current[18][3] = {0};    // HSV当前帧颜色 
    uint8_t *grey_img = NULL; // 75KB
    uint8_t *rgb565_out_buf = NULL; // 150KB
    int64_t time_start = 0;
    bool run_solve_local = true; // 上位机计算还是本地计算还原方式，默认本地计算
    int run_solve_local_char_cnt = 0;
    const int64_t TIME_OUT = 5000000;// 5s
    for(int i=0; i<54; i++){
        g_cube_str[i] = '?';
    }
    g_cube_str[54] = '\0';
    // 性能测试
    if(cfg.performance_test){
        run_solve_test("/spiffs/testcase.txt");
    }
    // 初始化摄像头
    ESP_LOGI(TAG, "INIT_CAMERA");
    if(ESP_OK != init_camera(true, ae_level)) {
        return;
    } 
    // 安装 USB Serial JTAG 驱动
    usb_serial_jtag_driver_config_t config = {
        .rx_buffer_size = 512,
        .tx_buffer_size = 128
    };
    usb_serial_jtag_driver_install(&config);
    // 主循环
    while(true){    
        // 以非阻塞方式查询调试串口有没有收到‘#’
        // 连续收到3个‘#’，进入上位机计算模式
        // 并且通过ESP_LOGI输出特定的字符串，通知上位机
        // getchar是阻塞的，用不了，需要使用底层的usb_serial_jtag_read_bytes函数
        int ch;
        if(run_solve_local && usb_serial_jtag_read_bytes(&ch, 1, 0) == 1){
            ESP_LOGI(TAG, "usb_serial_jtag_read_bytes: %c", ch);
            if(ch == '#'){
                if(++run_solve_local_char_cnt == 3){
                    ESP_LOGI(TAG, "enter host run_solve mode");
                    run_solve_local = false;
                }
            }else{
                run_solve_local_char_cnt = 0;
            }
        }
        if(ae_level != cfg.camera_ae_level) {
            ae_level = cfg.camera_ae_level;
            ESP_LOGI(TAG, "change camera AE level to %d", ae_level);
            init_camera(false, ae_level);
        }
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "esp_camera_fb_get failed");
            continue;
        }
        
        // 检查队列是否已满，如果没有满，复制一份放入队列，用于web实时预览
        if (uxQueueSpacesAvailable(jpeg_queue) != 0) {
            jpeg_buf jpg_bug;
            jpg_bug.buf = (uint8_t *)malloc(fb->len);
            jpg_bug.len = fb->len;
            memcpy(jpg_bug.buf, fb->buf, fb->len);
            xQueueSend(jpeg_queue, &jpg_bug, 0);
        }
        // 如果没有开启电机控制，不进行图像处理
        if(atomic_load(&g_motor_enable) != true){
            esp_camera_fb_return(fb);
            continue;
        }
        // 解压JPEG
        int width = fb->width;
        int height = fb->height;
        jpeg_error_t ret = jpeg_decode_one_frame(fb, &rgb565_out_buf);

        // 释放摄像头驱动的帧缓冲区
        esp_camera_fb_return(fb);
        if(ret != JPEG_ERR_OK){
            ESP_LOGE(TAG, "jpeg_decode_one_frame failed, error code = %d", ret);
            continue;
        }
        if(state == STATE_INIT){
            // 舍弃前10帧，让摄像头稳定
            if (++frame_counter >= cfg.background_discard_frames){
                state = STATE_BACKGROUND;
                frame_counter = 0;
                ESP_LOGI(TAG, "进入背景色采集状态");
            }
        }else if(state == STATE_BACKGROUND){
            // 记录背景颜色
            get_color(rgb565_out_buf, width, height, color_data_background);
            update_g_color_data(FACE_1, color_data_background);
            ESP_LOGI(TAG, "已采集背景色");
            state = STATE_DETECTING;
        }else if(state == STATE_DETECTING){
            if(cfg.auto_cube_detect != CUBE_DETECT_MANUAL){
                // 解除堵转报错
                if(atomic_load(&g_motor_error) == true){
                    atomic_store(&g_motor_error, false);
                    motion_cmd_t cmd = CMD_GO_HOME;
                    xQueueSend(motion_cmd_queue, &cmd, 0);
                    continue;
                }
                // 检测魔方放置
                get_color(rgb565_out_buf, width, height, color_data_current);
                update_g_color_data(FACE_1, color_data_current);
                // 计算变化量
                int changed_blocks = 0;
                for(int i=0; i<18; i++){
                    uint8_t diff = hs_distanse(color_data_current[i][0], color_data_current[i][1], 
                                            color_data_background[i][0], color_data_background[i][1]);
                    if (diff > cfg.sat_hue_threshold){
                        changed_blocks += 1;
                    }
                }
                if (changed_blocks >= cfg.change_threshold){
                    // 记录初始灰度图
                    int total_pixels = width * height;
                    if(grey_img == NULL){
                        grey_img = heap_caps_malloc(total_pixels, MALLOC_CAP_SPIRAM);// 75KB
                    }
                    if(grey_img == NULL){
                        ESP_LOGE(TAG, "Failed to malloc memory for grey_img");
                    }else{
                        ESP_LOGI(TAG, "差异色块数量%d, 检测到魔方放置, 等待图像稳定", changed_blocks);
                        frame_counter = 0;
                        const uint16_t *pixel_ptr = (uint16_t *)rgb565_out_buf;
                        for(int i = 0; i< total_pixels; i++){
                            grey_img[i] = rgb565_to_grey(pixel_ptr[i]);
                        }
                        time_start = esp_timer_get_time();
                        atomic_store(&g_time_cost_us, 0);
                        state = STATE_STABILIZING;
                    }
                }
            }else{
                // 手动模式，不检测魔方放置，只识别颜色
                get_color(rgb565_out_buf, width, height, color_data_current);
                update_g_color_data(FACE_1, color_data_current);
                // 手动触发
                if(check_restore_key_press()){
                    ESP_LOGI(TAG, "手动触发魔方还原");
                    // 记录当前的色块颜色，在STATE_CAPTURE_FACE_1再采集一次，确保采集结果不受图像模糊的影响
                    memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
                    frame_counter = 1;
                    state = STATE_CAPTURE_FACE_1;
                    // 以开始魔方颜色采集为节点，开始计算耗时
                    time_start = esp_timer_get_time();
                }
            }
        }else if(state == STATE_STABILIZING){
            // 这两条仅用于web预览功能，不影响整体流程
            get_color(rgb565_out_buf, width, height, color_data_current);
            update_g_color_data(FACE_1, color_data_current);

            int change_pixels = 0;
            int thresh_diff = cfg.thresh_diff;
            const uint16_t *pixel_ptr = (uint16_t *)rgb565_out_buf;
            int total_pixels = width * height;
            // 对比灰度图差异，同时将当前的图像转换为灰度图，供下一次对比
            // 为了性能，此处不提取色块颜色
            for(int i = 0; i< total_pixels; i++){
                uint8_t grey = rgb565_to_grey(pixel_ptr[i]);
                if(abs(grey - grey_img[i]) > thresh_diff){
                    change_pixels++;
                }
                grey_img[i] = grey;
            }
            float change_percent = (float)change_pixels / (float)total_pixels * 100.0f;
            if(change_percent < cfg.stabilization_percent){
                frame_counter++;
            }else{
                frame_counter=0;
            }
            if(frame_counter >= cfg.stabilization_frames){
                ESP_LOGI(TAG, "图像已经稳定%d帧，目前变化像素比例%.5f%%，开始魔方颜色识别", frame_counter, change_percent);
                // 记录当前的色块颜色，在STATE_CAPTURE_FACE_1再采集一次，确保采集结果不受图像模糊的影响
                memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
                update_g_color_data(FACE_1, color_data_prev);
                frame_counter = 1;
                state = STATE_CAPTURE_FACE_1;
                // 以开始魔方颜色采集为节点，开始计算耗时
                time_start = esp_timer_get_time();
            }
            if(esp_timer_get_time() - time_start > TIME_OUT){
                ESP_LOGI(TAG, "图像已经稳定%d帧，目前变化像素比例%.5f%%，等待超时，重新检测魔方放置", frame_counter, change_percent);
                frame_counter = 0;
                state = STATE_DETECTING;
            }
        }else if(state == STATE_CAPTURE_FACE_1){
            get_color(rgb565_out_buf, width, height, color_data_current);
            update_g_color_data(FACE_1, color_data_current);
            int equal_count = 0;
            for(int i=0; i<18; i++){
                if(hs_equal(color_data_current[i][0], color_data_current[i][1], color_data_prev[i][0], color_data_prev[i][1], cfg.hs_equal_level)){
                    equal_count++;
                }
            }
            frame_counter++;
            if(equal_count == 18){
                ESP_LOGI(TAG, "时间点%.3fs, 完成第1组颜色信息的采集，共对比图像%d帧", 1e-6f * (esp_timer_get_time() - time_start), frame_counter);
                frame_counter = 0;
                // 发送魔方翻转指令
                atomic_store(&g_flip_completed, false);
                motion_cmd_t cmd = CMD_FLIP_TO_FACE_2;
                xQueueSend(motion_cmd_queue, &cmd, 0);
                state = STATE_FLIP_TO_FACE_2;
            }else{
                ESP_LOGI(TAG, "equal_count = %d", equal_count);
                memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
            }
            atomic_store(&g_time_cost_us, (int)(esp_timer_get_time() - time_start));
        }else if(state == STATE_FLIP_TO_FACE_2){
            get_color(rgb565_out_buf, width, height, color_data_current);
            update_g_color_data(FACE_2, color_data_current);
            if(atomic_load(&g_flip_completed) == true){
                 memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
                 frame_counter = 1;
                 state = STATE_CAPTURE_FACE_2;
            }
            atomic_store(&g_time_cost_us, (int)(esp_timer_get_time() - time_start));
        }else if(state == STATE_CAPTURE_FACE_2){
            get_color(rgb565_out_buf, width, height, color_data_current);
            update_g_color_data(FACE_2, color_data_current);
            int equal_count = 0;
            for(int i=0; i<18; i++){
                if(hs_equal(color_data_current[i][0], color_data_current[i][1], color_data_prev[i][0], color_data_prev[i][1], cfg.hs_equal_level)){
                    equal_count++;
                }
            }
            frame_counter++;
            if(equal_count == 18){
                ESP_LOGI(TAG, "时间点%.3fs, 完成第2组颜色信息的采集，共对比图像%d帧", 1e-6f * (esp_timer_get_time() - time_start), frame_counter);
                frame_counter = 0;
                // 发送魔方翻转指令
                atomic_store(&g_flip_completed, false);
                motion_cmd_t cmd = CMD_FLIP_TO_FACE_3;
                xQueueSend(motion_cmd_queue, &cmd, 0);
                state = STATE_FLIP_TO_FACE_3;
            }else{
                ESP_LOGI(TAG, "equal_count = %d", equal_count);
                memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
            }
            atomic_store(&g_time_cost_us, (int)(esp_timer_get_time() - time_start));
        }else if(state == STATE_FLIP_TO_FACE_3){
            get_color(rgb565_out_buf, width, height, color_data_current);
            update_g_color_data(FACE_3, color_data_current);
            if(atomic_load(&g_flip_completed) == true){
                 memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
                 frame_counter = 1;
                 state = STATE_CAPTURE_FACE_3;
            }
            atomic_store(&g_time_cost_us, (int)(esp_timer_get_time() - time_start));
        }else if(state == STATE_CAPTURE_FACE_3){
            get_color(rgb565_out_buf, width, height, color_data_current);
            update_g_color_data(FACE_3, color_data_current);
            int equal_count = 0;
            for(int i=0; i<18; i++){
                if(hs_equal(color_data_current[i][0], color_data_current[i][1], color_data_prev[i][0], color_data_prev[i][1], cfg.hs_equal_level)){
                    equal_count++;
                }
            }
            frame_counter++;
            if(equal_count == 18){
                ESP_LOGI(TAG, "时间点%.3fs, 完成第3组颜色信息的采集，共对比图像%d帧", 1e-6f * (esp_timer_get_time() - time_start), frame_counter);
                frame_counter = 0;
                char white_green_pos[3];
                if(color_detect(g_color_data, g_cube_str, white_green_pos) == 0){
                    ESP_LOGI(TAG, "魔方图像识别结果：%s 白中心块：%c 绿中心块：%c", g_cube_str, white_green_pos[0], white_green_pos[1]);
                    if(strcmp(g_cube_str, SOLVED_CUBE) == 0){
                        // 计算打乱方式
                        g_estimate_time.motion_sequence = run_scramble(g_motion_sequence, cfg.scramble_sequence, white_green_pos);
                    }else{
                        if(run_solve_local) {
                            // 本地计算模式
                            g_estimate_time.motion_sequence = run_solve(g_cube_str, g_motion_sequence, sizeof(g_motion_sequence));
                        } else {
                            // 通知上位机魔方状态
                            ESP_LOGI(TAG, "run_solve = %s", g_cube_str);
                            if(receive_host_command(g_motion_sequence, &g_estimate_time.motion_sequence, sizeof(g_motion_sequence)) != 0) {
                                // 接收失败，使用空指令
                                g_motion_sequence[0] = '\0';
                                g_estimate_time.motion_sequence = 0;
                            }
                        }
                    }
                    if(cfg.auto_cube_detect != CUBE_DETECT_ONLY_CAPTURE_FACE){
                        // 发送魔方还原指令
                        atomic_store(&g_flip_completed, false);
                        motion_cmd_t cmd = CMD_MOTIONS;
                        xQueueSend(motion_cmd_queue, &cmd, 0);
                    }else{
                        atomic_store(&g_flip_completed, false);
                        motion_cmd_t cmd = CMD_RELEASE_CUBE;
                        xQueueSend(motion_cmd_queue, &cmd, 0);
                    }
                }else{
                    atomic_store(&g_flip_completed, false);
                    motion_cmd_t cmd = CMD_RELEASE_CUBE;
                    xQueueSend(motion_cmd_queue, &cmd, 0);
                }
                
                state = STATE_SOLVE;
            }else{
                ESP_LOGI(TAG, "equal_count = %d", equal_count);
                memcpy(color_data_prev, color_data_current, sizeof(color_data_current));
            }
            atomic_store(&g_time_cost_us, (int)(esp_timer_get_time() - time_start));
        }else if(state == STATE_SOLVE){
            if(atomic_load(&g_flip_completed) == true){
                ESP_LOGI(TAG, "时间点%.3fs, 完成还原！", 1e-6f * (esp_timer_get_time() - time_start));
                if(cfg.auto_cube_detect != CUBE_DETECT_MANUAL){
                    state = STATE_DETECTING_REMOVE;
                }else{
                    // 清空已保存的魔方颜色识别结果
                    memset(g_color_data, 0, sizeof(g_color_data));
                    for(int i=0; i<54; i++){
                        g_cube_str[i] = '?';
                    }
                    g_cube_str[54] = '\0';
                    state = STATE_DETECTING;
                }
            }
            atomic_store(&g_time_cost_us, (int)(esp_timer_get_time() - time_start));
        }else if(state == STATE_DETECTING_REMOVE){
            // 检测魔方放置
            get_color(rgb565_out_buf, width, height, color_data_current);
            // update_g_color_data(FACE_1, color_data_current);
            // 计算变化量
            int changed_blocks = 0;
            for(int i=0; i<18; i++){
                uint8_t diff = hs_distanse(color_data_current[i][0], color_data_current[i][1], 
                                           color_data_background[i][0], color_data_background[i][1]);
                if (diff > cfg.sat_hue_threshold){
                    changed_blocks += 1;
                }
            }
            if (changed_blocks <= cfg.change_threshold_remove){
                ESP_LOGI(TAG, "差异色块数量%d, 检测到魔方移除", changed_blocks);
                // 清空已保存的魔方颜色识别结果
                memset(g_color_data, 0, sizeof(g_color_data));
                for(int i=0; i<54; i++){
                    g_cube_str[i] = '?';
                }
                g_cube_str[54] = '\0';
                state = STATE_DETECTING;
            }
        }
        // 更新状态信息，用于web端显示
        atomic_store(&g_camera_state, state);
        // 短暂释放CPU，可调度其他低优先级任务
        vTaskDelay(1);
    }
    // 分配grey_img，rgb565_out_buf后不再释放，一直复用同一片内存，实际不会执行以下语句
    free(grey_img); 
    free(rgb565_out_buf);
}
// -------------------------------------- main -------------------------------------- 
void app_main(void)
{
    bool config_error = false;
    cfg.sta_ssid[0] = '\0';
    cfg.sta_password[0] = '\0';
    // --------------------------- 加载储存在FLASH中的参数 ---------------------------
    mount_spiffs();
    if(load_motion_cfg(MOTION_CFG_PATH) != 0){
        config_error = true;
    }
    if(load_main_cfg(MAIN_CFG_PATH) != 0){
        config_error = true;
    }
    // --------------------------- 初始化WIFI热点 ---------------------------
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    // wifi_init_softap();
    wifi_init_sta_ap();

    // --------------------------- 初始化需要使用的低速外设---------------------------
    init_pwm();
    init_rs485();
    init_gpio();

    // --------------------------- 初始化WEB服务器 ---------------------------
    // 运动控制指令队列初始化，可存储4条指令
    motion_cmd_queue = xQueueCreate(4, sizeof(motion_cmd_t));
    if (motion_cmd_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create motion command queue");
        return;
    }
    // jpeg传输队列
    jpeg_queue = xQueueCreate(2, sizeof(jpeg_buf));
    if (motion_cmd_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create jpeg queue");
        return;
    }
    setup_servers();


    // 从FLASH中加载color_detect_points，并且重新计算color_detect_quads
    update_quads_info();
    update_led_brightness(cfg.led_brightness);
    init_time_table(&get_time_motions_str);
    update_camera_flip_time();
    // --------------------------- 创建任务 ---------------------------
    if(!config_error){
        // 创建绑定到核心1的高优先级任务
        xTaskCreatePinnedToCore(
            motion_task,           // 任务函数
            "motion_task",         // 任务名称
            8192,                  // 任务栈大小(字节)
            NULL,                  // 传递参数
            configMAX_PRIORITIES-3,// 优先级（最高-3）
            NULL,                  // 任务句柄(可选)
            1                      // 核心编号(0或1)
        );
        xTaskCreatePinnedToCore(
            camera_task,           // 任务函数
            "camera_task",         // 任务名称
            8192,                  // 任务栈大小(字节)
            NULL,                  // 传递参数
            configMAX_PRIORITIES-5,// 优先级（最高-5）
            NULL,                  // 任务句柄(可选)
            1                      // 核心编号(0或1)
        );
    }
    int64_t time_start = esp_timer_get_time();
    while(true){
        int status = atomic_load(&g_camera_state);
        if(status == STATE_DETECTING){
            gpio_set_level(GPIO_DBG_LED, 1);
            vTaskDelay(pdMS_TO_TICKS(250));
            gpio_set_level(GPIO_DBG_LED, 0);
            vTaskDelay(pdMS_TO_TICKS(250));
        }else{
            gpio_set_level(GPIO_DBG_LED, 1);
            vTaskDelay(pdMS_TO_TICKS(500));
        }
        // 获取历史最小空闲堆
        size_t min_sram = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        size_t min_psram = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
        int64_t time_now = esp_timer_get_time();
        if(time_now - time_start > 30 * 1000000){
            time_start = time_now;
            ESP_LOGI(TAG, "历史最小剩余内存，PSRAM %.2fKB， SRAM %.2fKB", min_psram/1024.0f, min_sram/1024.0f);
        }
    }
}