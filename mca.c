#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "nvs_flash.h"

static const char *TAG = "CSI_TX";

// A raw 802.11 Action frame payload (Broadcast)
// This is exactly what the Receiver ESP32 will intercept to calculate CSI.
static uint8_t raw_frame[] = {
    // Frame Control: 0x00D0 (Action frame)
    0xd0, 0x00, 
    // Duration
    0x00, 0x00, 
    // Destination MAC (Broadcast)
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
    // Source MAC (Will be overwritten by the Wi-Fi driver)
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 
    // BSSID 
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 
    // Sequence Control
    0x00, 0x00, 
    // Payload (Vendor-specific action frame data)
    0xdd, 0x07, 
    0x18, 0xfe, 0x34, 0x00, 0x01, 0x02, 0x03 
};

// FreeRTOS Task to continuously broadcast frames
void tx_task(void *pvParameter)
{
    ESP_LOGI(TAG, "Starting to broadcast raw frames on Channel 6...");
    
    while (1) {
        // Inject the raw frame directly into the air
        esp_err_t err = esp_wifi_80211_tx(ESP_IF_WIFI_STA, raw_frame, sizeof(raw_frame), true);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to send frame: %s", esp_err_to_name(err));
        }
        
        // Delay 10ms -> Broadcast rate of 100 Hz. 
        // 100 Hz is the standard sampling rate for detecting human movement and respiration.
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
}

void app_main(void)
{
    // 1. Initialize NVS (Required for the Wi-Fi driver to function)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Initialize the core Wi-Fi stack
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 3. Set to Station Mode and start the driver
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // 4. Lock Wi-Fi to a specific channel (e.g., Channel 6)
    // CRITICAL: The Receiver ESP32 must be locked to this exact same channel!
    ESP_ERROR_CHECK(esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE));

    // 5. Spawn the transmission task on a separate FreeRTOS thread
    xTaskCreate(tx_task, "tx_task", 4096, NULL, 5, NULL);
}