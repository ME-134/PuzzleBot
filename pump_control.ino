#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>

#define V1 A0
#define V2 A1
#define PUMP 12

ros::NodeHandle node_handle;

std_msgs::Float32 current_msg;
std_msgs::UInt16 pump_msg;

void pump_cb(const std_msgs::UInt16& pump_msg) {
  if (pump_msg.data  == 1) {
    digitalWrite(PUMP, HIGH); 
  } else {
    digitalWrite(PUMP, LOW);
  }
}

ros::Publisher current_publisher("current_draw", &current_msg);
ros::Subscriber<std_msgs::UInt16> led_subscriber("toggle_pump", &pump_cb);

void setup()
{
  pinMode(PUMP, OUTPUT);
  pinMode(BUTTON, INPUT);
  
  node_handle.initNode();
  node_handle.advertise(current_publisher);
  node_handle.subscribe(led_subscriber);
}

void loop()
{ 
  current_msg.data = analogRead(V2) - analogRead(V1);

  current_publisher.publish( &current_msg );
  node_handle.spinOnce();
  
  delay(100);
}