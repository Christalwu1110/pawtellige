class AbnormalBehaviorDetector:
    def __init__(self, logger):
        self.logger = logger
        self.confidence_threshold = 0.65
        self.inactive_duration = 300
        self.eating_duration = 180

    def check_for_abnormalities(self, current_check_time):
        records = self.logger.get_records()
        if not records:
            return
        
        updated_records = []
        for record in records:
            new_record = record.copy()
            new_record['abnormal_flags'] = new_record.get('abnormal_flags', [])
            
            if new_record['confidence'] < self.confidence_threshold:
                if "Low confidence" not in new_record['abnormal_flags']:
                    new_record['abnormal_flags'].append("Low confidence")
            
            updated_records.append(new_record)
        
        self.logger.update_records(updated_records)
        